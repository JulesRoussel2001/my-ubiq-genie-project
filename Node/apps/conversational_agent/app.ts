// VERSION: Tone + 5-way congruence guardrail (cue-conditional: only apply when a cue is detected)
import { NetworkId } from "ubiq";
import { ApplicationController } from "../../components/application";
import { TextToSpeechService } from "../../services/text_to_speech/service";
import { TextGenerationService } from "../../services/text_generation/service";
import { AnimationsService } from "../../services/animations/service";
import { MessageReader } from "../../components/message_reader";
import path from "path";
import { fileURLToPath } from "url";

/** Wire constants */
const SR = 48000;
const BYTES_PER_SECOND = 2 * SR;   // 16-bit mono 48 kHz
const CHUNK = 16000;               // bytes per packet
const AUDIO_CH = 95;               // header + PCM + ACK live here

type AckWaiter = { resolve:(v:boolean)=>void, timeout:NodeJS.Timeout };

/* ------------------------ Guardrail config (frozen) ------------------------ */

/** 5-way label space */
type Label5 = "pos" | "sad" | "ang" | "unc" | "neu";

/** Cue lists (lightweight; frozen) */
const POS_RE = /\b(hello|great|awesome|congrats|congratulations|well done|glad|happy|nice|love|fantastic|excellent|thanks|thank you|bravo|amazing|wonderful|cool|sweet|yay)\b/i;
const SAD_RE = /\b(sorry|sad|upset|unhappy|hurt|regret|pain|heartbroken|depressed|down)\b/i;
const ANG_RE = /\b(angry|mad|furious|annoyed|frustrat|disapprove|outrage|irritat)\b/i;
const UNC_RE = /\b(worried|anxious|nervous|afraid|scared|unsure|not sure|confus|uncertain|what\?|really\?|no way|surpris)\b/i;

/** Allowed animation sets per 5-way label (priority order; keep Talking as a fallback) */
const ALLOWED_5: Record<Label5, string[]> = {
  pos: ["Happy Hand Gesture","Happy Idle","Laughing","Clapping","Waving","Talking"],
  sad: ["Sad Idle","Look Away Gesture","Shrugging","Talking"],
  ang: ["Angry Gesture","Disgust Gesture","Look Away Gesture","Shrugging","Talking"],
  unc: ["Fear Gesture","Surprise Gesture","Look Around","Look Away Gesture","Shrugging","Talking","Waving"],
  neu: ["Talking","Look Around","Shrugging","Happy Idle"],
};

/** Animation → Azure TTS style (conservative, with safe fallbacks in Python worker) */
const animToStyle: Record<string,string> = {
  "Happy Hand Gesture": "cheerful",
  "Laughing": "cheerful",
  "Clapping": "cheerful",
  "Waving": "cheerful",
  "Happy Idle": "cheerful",

  "Angry Gesture": "angry",
  "Disgust Gesture": "angry",      // proxy for disapproval/aversion (low degree recommended)
  "Sad Idle": "sad",

  "Fear Gesture": "terrified",
  "Surprise Gesture": "excited",

  "Look Around": "calm",
  "Shrugging": "calm",
  "Look Away Gesture": "calm",     // aversion/disapproval proxy
  "Talking": "neutral",
};

/** Our animation vocabulary (whitelist) */
const KNOWN_ANIMS = Object.keys(animToStyle);

/* ------------------------ App ------------------------ */

export class ConversationalAgent extends ApplicationController {
  components: {
    chatReader?: MessageReader;            // 97 UI input
    ackReader?: MessageReader;             // 95 back-ACKs from Unity
    textGenerationService?: TextGenerationService;
    textToSpeechService?: TextToSpeechService;
    animationsService?: AnimationsService;
  } = {};

  private busy = false;
  private turnQueue: Array<{ msg: string; targetPeer: string }> = [];
  private seq = 0;
  private targetPeer = "default";
  private ackWaiters: Map<number,AckWaiter> = new Map();

  constructor(configFile: string = "config.json") { super(configFile); }

  start(): void {
    this.registerComponents();
    this.joinRoom().then(() => this.definePipeline());
  }

  registerComponents() {
    this.components.textGenerationService = new TextGenerationService(this.scene);
    this.components.textToSpeechService   = new TextToSpeechService(this.scene);
    this.components.animationsService     = new AnimationsService(this.scene);
    this.components.chatReader            = new MessageReader(this.scene, 97);
    this.components.ackReader             = new MessageReader(this.scene, AUDIO_CH);
  }

  // ----------------- helpers -----------------
  private splitSentences(text: string): string[] {
    const cleaned = (text || "").replace(/\s+/g, " ").trim();
    if (!cleaned) return [];
    const parts = cleaned.match(/[^.!?]+[.!?]*/g) || [];
    return parts.map(s => s.trim()).filter(Boolean);
  }

  private onceFromService(
    svc: any,
    _label: string,
    send: () => void,
    timeoutMs = 20000
  ): Promise<Buffer> {
    return new Promise((resolve, reject) => {
      let settled = false;
      const finish = (ok: boolean, payload?: Buffer | any) => {
        if (settled) return;
        settled = true;
        clearTimeout(to);

        const off = (ev:string, fn:any)=>{ svc.off?.(ev, fn); svc.removeListener?.(ev, fn); };
        off("response", onResponse);
        off("data", onData);
        off("error", onErr);

        if (ok) {
          resolve(Buffer.isBuffer(payload) ? payload : Buffer.from(payload ?? ""));
        } else {
          reject(new Error("service error"));
        }
      };
      const onResponse = (b: Buffer) => finish(true, b);
      const onData     = (b: Buffer) => finish(true, b);
      const onErr      = (_e: any)    => finish(false);
      const to = setTimeout(() => finish(false), timeoutMs);

      svc.on?.("response", onResponse);
      svc.on?.("data", onData);
      svc.on?.("error", onErr);

      send();
    });
  }

  /** Normalize any animation string from the service to our whitelist (strict) */
  private normalizeAnim(raw: string): string {
    if (!raw) return "Talking";
    const s = raw.toLowerCase().replace(/[^a-z\s]/g,"").trim();

    // exact match (case-insensitive)
    for (const k of KNOWN_ANIMS) {
      if (k.toLowerCase() === s) return k;
    }
    // contains match on last token (e.g., "maybe try waving" -> "Waving")
    for (const k of KNOWN_ANIMS) {
      const tail = k.toLowerCase().split(" ").pop()!;
      if (tail && s.includes(tail)) return k;
    }
    // default safe
    return "Talking";
  }

  /** Detect an affect cue; return a label only if a cue is found, else null (no default "neu"). */
  private detectCue(sentence: string): Label5 | null {
    const s = (sentence || "").toLowerCase();

    const hitAng = ANG_RE.test(s);
    const hitSad = SAD_RE.test(s);
    const hitPos = POS_RE.test(s);
    const hitUnc = UNC_RE.test(s);

    // Priority: explicit anger > sadness > positive > uncertainty
    if (hitAng) return "ang";
    if (hitSad) return "sad";
    if (hitPos) return "pos";
    if (hitUnc) return "unc";
    return null; // no cue → do not force a label
  }

  /** Apply 5-way guardrail: if anim not allowed for this label, pick the first allowed. */
  private enforceGuardrail5(label: Label5, anim: string): string {
    const allowed = ALLOWED_5[label] || ALLOWED_5.neu;
    return allowed.includes(anim) ? anim : (allowed[0] || "Talking");
  }

  /** TTS once (JSON in, framed bytes out): expects "LEN:<n>\n" then n PCM bytes */
  private ttsFramedOnce(sentence: string, style: string = "neutral", degree = 1.0, timeoutMs=25000): Promise<Buffer> {
    return new Promise((resolve, reject) => {
      const tts:any = this.components.textToSpeechService!;
      let mode:"header"|"body" = "header";
      let header = Buffer.alloc(0), body = Buffer.alloc(0);
      let remaining = 0;

      const onData = (raw:Buffer) => {
        let data = Buffer.isBuffer(raw) ? raw : Buffer.from(raw as any);
        while (data.length) {
          if (mode === "header") {
            header = Buffer.concat([header, data]);
            const nl = header.indexOf(0x0A);
            if (nl === -1) return;
            const line = header.slice(0,nl).toString("utf8").trim();
            data = header.slice(nl+1);
            header = Buffer.alloc(0);
            if (!line.startsWith("LEN:")) return;
            remaining = parseInt(line.slice(4).trim(),10) || 0;
            body = Buffer.alloc(0); mode = "body";
          } else {
            const take = Math.min(remaining, data.length);
            if (take) {
              body = Buffer.concat([body, data.slice(0,take)]);
              remaining -= take; data = data.slice(take);
            }
            if (remaining === 0) {
              cleanup();
              resolve(body);
              return;
            }
          }
        }
      };
      const onErr = (_e:any)=>{ cleanup(); reject(new Error("tts error")); };
      const to = setTimeout(()=>{ cleanup(); reject(new Error("tts timeout")); }, timeoutMs);
      const cleanup = ()=> {
        clearTimeout(to);
        tts.off?.("data", onData); tts.off?.("error", onErr);
        tts.removeListener?.("data", onData); tts.removeListener?.("error", onErr);
      };

      tts.on?.("data", onData);
      tts.on?.("error", onErr);

      // JSON payload -> Python child builds SSML with style (and falls back if unsupported)
      const payload = JSON.stringify({ text: sentence, style, degree });
      tts.sendToChildProcess("default", payload + "\n");
    });
  }

  private async llmOnce(prompt: string): Promise<string> {
    const svc: any = this.components.textGenerationService!;
    const buf = await this.onceFromService(svc, "LLM", () =>
      svc.sendToChildProcess("default", prompt + "\n"),
      30000
    );
    let s = buf.toString();
    if (s.startsWith(">")) s = s.slice(1);
    return s.replace(/(\r\n|\n|\r)/g, " ").trim();
  }

  private async animOnce(sentence: string): Promise<string> {
    const svc: any = this.components.animationsService!;
    const buf = await this.onceFromService(svc, "ANIM", () =>
      svc.sendToChildProcess("default", sentence + "\n"),
      15000
    ).catch(() => Buffer.from("Talking"));
    const nameRaw = (Buffer.isBuffer(buf) ? buf.toString() : String(buf)).trim();
    return this.normalizeAnim(nameRaw);
  }

  private sendHeader(seq:number, anim:string, pcmLen:number) {
    const header = {
      type: "A",
      seq,
      targetPeer: this.targetPeer,
      audioLength: String(pcmLen),
      animationTitle: anim || "Talking",
      sampleRate: SR,
    };
    this.scene.send(new NetworkId(AUDIO_CH), header);
  }

  private streamPcm(pcm:Buffer) {
    for (let rest = pcm; rest.length; rest = rest.slice(CHUNK)) {
      const piece = rest.slice(0, CHUNK);
      this.scene.send(new NetworkId(AUDIO_CH), piece);
    }
  }

  /** wait for ACK: resolves true on ACK, false on timeout */
  private waitAck(seq:number, expectMs:number): Promise<boolean> {
    const ACK_MIN = 1000;
    const ACK_MAX = 15000;
    const timeoutMs = Math.min(ACK_MAX, Math.max(ACK_MIN, Math.floor(expectMs * 1.3) + 250));

    return new Promise<boolean>((resolve) => {
      const timer = setTimeout(() => {
        this.ackWaiters.delete(seq);
        resolve(false);
      }, timeoutMs);
      this.ackWaiters.set(seq, { resolve, timeout: timer });
    });
  }

  private handleAck(jsonStr:string) {
    try {
      const msg = JSON.parse(jsonStr);
      if (msg && msg.type === "SentenceDone" && typeof msg.seq === "number") {
        const waiter = this.ackWaiters.get(msg.seq);
        if (waiter) {
          clearTimeout(waiter.timeout);
          this.ackWaiters.delete(msg.seq);
          waiter.resolve(true);
        }
      }
    } catch { /* ignore */ }
  }

  // ----------------- turns -----------------
  private enqueueTurn(msg:string, targetPeer:string){
    this.turnQueue.push({msg, targetPeer});
    if (!this.busy) this.processTurns();
  }

  private async processTurns() {
    this.busy = true;
    while (this.turnQueue.length) {
      const turn = this.turnQueue.shift()!;
      this.targetPeer = turn.targetPeer;

      const reply = await this.llmOnce(turn.msg);
      const sentences = this.splitSentences(reply);

      for (const sentence of sentences) {
        const seq = ++this.seq;

        // 1) LLM proposes animation (service) → normalize
        const animSuggestion = await this.animOnce(sentence);
        const animNorm = this.normalizeAnim(animSuggestion);

        // 2) Detect affect cue (or null). No default "neu".
        const cue = this.detectCue(sentence);

        // 3) Guardrail: apply ONLY when a cue is detected; else keep LLM's choice
        const anim = cue ? this.enforceGuardrail5(cue, animNorm) : animNorm;

        // 4) TTS with style derived from animation (worker handles unsupported fallbacks)
        const style = animToStyle[anim] || "neutral";
        const pcm  = await this.ttsFramedOnce(sentence, style, 0.9);

        // 5) Send header + stream PCM + await ACK
        this.sendHeader(seq, anim, pcm.length);
        await new Promise(r=>setImmediate(r)); // let Unity parse header first
        this.streamPcm(pcm);

        const ms = Math.ceil(1000 * pcm.length / BYTES_PER_SECOND);
        await this.waitAck(seq, ms);
      }
    }
    this.busy = false;
  }

  definePipeline() {
    // UI input (channel 97)
    this.components.chatReader!.on("data", (data:any) => {
      try {
        const raw = data.message?.toString?.();
        const payload = JSON.parse(raw);
        const msg = (payload.message || "").trim();
        const targetPeer = payload.targetPeer || "default";
        if (!msg) return;
        this.enqueueTurn(msg, targetPeer);
      } catch {
        // ignore malformed UI packets
      }
    });

    // ACK listener (same channel 95)
    this.components.ackReader!.on("data", (data:any) => {
      const buf = data.message;
      if (!buf || !buf.length) return;
      if (buf[0] !== "{".charCodeAt(0)) return; // ACKs are JSON
      this.handleAck(buf.toString());
    });
  }
}

if (fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const absConfig = path.resolve(__dirname, "./config.json");
  const app = new ConversationalAgent(absConfig);
  app.start();
}

// VERSION WITH TONE
// import { NetworkId } from "ubiq";
// import { ApplicationController } from "../../components/application";
// import { TextToSpeechService } from "../../services/text_to_speech/service";
// import { TextGenerationService } from "../../services/text_generation/service";
// import { AnimationsService } from "../../services/animations/service";
// import { MessageReader } from "../../components/message_reader";
// import path from "path";
// import { fileURLToPath } from "url";

// /** Wire constants */
// const SR = 48000;
// const BYTES_PER_SECOND = 2 * SR;   // 16-bit mono
// const CHUNK = 16000;               // bytes
// const AUDIO_CH = 95;               // header + PCM + ACK live here

// /** Animation → Azure TTS style */
// const animToStyle: Record<string,string> = {
//   "Happy Hand Gesture": "cheerful",
//   "Laughing": "cheerful",
//   "Clapping": "cheerful",
//   "Waving": "cheerful",
//   "Happy Idle": "cheerful",
//   "Angry Gesture": "angry",
//   "Sad Idle": "sad",
//   "Look Around": "calm",
//   "Shrugging": "calm",
//   "Look Away Gesture": "embarrassed", // may fallback to neutral on some voices
//   "Talking": "neutral",
// };

// type AckWaiter = { resolve:(v:boolean)=>void, timeout:NodeJS.Timeout };

// export class ConversationalAgent extends ApplicationController {
//   components: {
//     chatReader?: MessageReader;            // 97 UI input
//     ackReader?: MessageReader;             // 95 back-ACKs from Unity
//     textGenerationService?: TextGenerationService;
//     textToSpeechService?: TextToSpeechService;
//     animationsService?: AnimationsService;
//   } = {};

//   private busy = false;
//   private turnQueue: Array<{ msg: string; targetPeer: string }> = [];
//   private seq = 0;
//   private targetPeer = "default";
//   private ackWaiters: Map<number,AckWaiter> = new Map();

//   constructor(configFile: string = "config.json") { super(configFile); }

//   start(): void {
//     this.registerComponents();
//     this.joinRoom().then(() => this.definePipeline());
//   }

//   registerComponents() {
//     this.components.textGenerationService = new TextGenerationService(this.scene);
//     this.components.textToSpeechService   = new TextToSpeechService(this.scene);
//     this.components.animationsService     = new AnimationsService(this.scene);
//     this.components.chatReader            = new MessageReader(this.scene, 97);
//     this.components.ackReader             = new MessageReader(this.scene, AUDIO_CH);
//   }

//   // ----------------- helpers -----------------
//   private splitSentences(text: string): string[] {
//     const cleaned = (text || "").replace(/\s+/g, " ").trim();
//     if (!cleaned) return [];
//     const parts = cleaned.match(/[^.!?]+[.!?]*/g) || [];
//     return parts.map(s => s.trim()).filter(Boolean);
//   }

//   private onceFromService(
//     svc: any,
//     _label: string,
//     send: () => void,
//     timeoutMs = 20000
//   ): Promise<Buffer> {
//     return new Promise((resolve, reject) => {
//       let settled = false;
//       const finish = (ok: boolean, payload?: Buffer | any) => {
//         if (settled) return;
//         settled = true;
//         clearTimeout(to);

//         const off = (ev:string, fn:any)=>{ svc.off?.(ev, fn); svc.removeListener?.(ev, fn); };
//         off("response", onResponse);
//         off("data", onData);
//         off("error", onErr);

//         if (ok) {
//           resolve(Buffer.isBuffer(payload) ? payload : Buffer.from(payload ?? ""));
//         } else {
//           reject(new Error("service error"));
//         }
//       };
//       const onResponse = (b: Buffer) => finish(true, b);
//       const onData     = (b: Buffer) => finish(true, b);
//       const onErr      = (_e: any)    => finish(false);
//       const to = setTimeout(() => finish(false), timeoutMs);

//       svc.on?.("response", onResponse);
//       svc.on?.("data", onData);
//       svc.on?.("error", onErr);

//       send();
//     });
//   }

//   /** Normalize any animation string coming back from the service to our whitelist. */
//   private normalizeAnim(raw: string): string {
//     if (!raw) return "Talking";
//     const s = raw.toLowerCase().replace(/[^a-z\s]/g,"").trim();
//     // exact match (case-insensitive)
//     for (const k of Object.keys(animToStyle)) {
//       if (k.toLowerCase() === s) return k;
//     }
//     // contains match (e.g., "i suggest waving" -> "Waving")
//     for (const k of Object.keys(animToStyle)) {
//       const key = k.toLowerCase().split(" ").pop()!; // last word
//       if (s.includes(key)) return k;
//     }
//     // simple sentiment buckets
//     if (/(happy|glad|great|awesome|thanks|smile|joy)/.test(s)) return "Happy Idle";
//     if (/(sad|sorry|unfortunately|regret|upset)/.test(s)) return "Sad Idle";
//     if (/(angry|mad|frustrat|annoy)/.test(s)) return "Angry Gesture";
//     if (/(laugh|funny|haha)/.test(s)) return "Laughing";
//     if (/(wave|hi|hello|bye)/.test(s)) return "Waving";
//     return "Talking";
//   }

//   /** TTS once (JSON in, framed bytes out): expects "LEN:<n>\n" then n PCM bytes on "data" */
//   private ttsFramedOnce(sentence: string, style: string = "neutral", degree = 1.0, timeoutMs=25000): Promise<Buffer> {
//     return new Promise((resolve, reject) => {
//       const tts:any = this.components.textToSpeechService!;
//       let mode:"header"|"body" = "header";
//       let header = Buffer.alloc(0), body = Buffer.alloc(0);
//       let remaining = 0;

//       const onData = (raw:Buffer) => {
//         let data = Buffer.isBuffer(raw) ? raw : Buffer.from(raw as any);
//         while (data.length) {
//           if (mode === "header") {
//             header = Buffer.concat([header, data]);
//             const nl = header.indexOf(0x0A);
//             if (nl === -1) return;
//             const line = header.slice(0,nl).toString("utf8").trim();
//             data = header.slice(nl+1);
//             header = Buffer.alloc(0);
//             if (!line.startsWith("LEN:")) return;
//             remaining = parseInt(line.slice(4).trim(),10) || 0;
//             body = Buffer.alloc(0); mode = "body";
//           } else {
//             const take = Math.min(remaining, data.length);
//             if (take) {
//               body = Buffer.concat([body, data.slice(0,take)]);
//               remaining -= take; data = data.slice(take);
//             }
//             if (remaining === 0) {
//               cleanup();
//               resolve(body);
//               return;
//             }
//           }
//         }
//       };
//       const onErr = (_e:any)=>{ cleanup(); reject(new Error("tts error")); };
//       const to = setTimeout(()=>{ cleanup(); reject(new Error("tts timeout")); }, timeoutMs);
//       const cleanup = ()=> {
//         clearTimeout(to);
//         tts.off?.("data", onData); tts.off?.("error", onErr);
//         tts.removeListener?.("data", onData); tts.removeListener?.("error", onErr);
//       };

//       tts.on?.("data", onData);
//       tts.on?.("error", onErr);

//       // Send JSON line so the Python worker can build SSML with style
//       const payload = JSON.stringify({ text: sentence, style, degree });
//       tts.sendToChildProcess("default", payload + "\n");
//     });
//   }

//   private async llmOnce(prompt: string): Promise<string> {
//     const svc: any = this.components.textGenerationService!;
//     const buf = await this.onceFromService(svc, "LLM", () =>
//       svc.sendToChildProcess("default", prompt + "\n"),
//       30000
//     );
//     let s = buf.toString();
//     if (s.startsWith(">")) s = s.slice(1);
//     return s.replace(/(\r\n|\n|\r)/g, " ").trim();
//   }

//   private async animOnce(sentence: string): Promise<string> {
//     const svc: any = this.components.animationsService!;
//     const buf = await this.onceFromService(svc, "ANIM", () =>
//       svc.sendToChildProcess("default", sentence + "\n"),
//       15000
//     ).catch(() => Buffer.from("Talking"));
//     const nameRaw = (Buffer.isBuffer(buf) ? buf.toString() : String(buf)).trim();
//     return this.normalizeAnim(nameRaw);
//   }

//   private sendHeader(seq:number, anim:string, pcmLen:number) {
//     const header = {
//       type: "A",
//       seq,
//       targetPeer: this.targetPeer,
//       audioLength: String(pcmLen),
//       animationTitle: anim || "Talking",
//       sampleRate: SR,
//     };
//     this.scene.send(new NetworkId(AUDIO_CH), header);
//   }

//   private streamPcm(pcm:Buffer) {
//     for (let rest = pcm; rest.length; rest = rest.slice(CHUNK)) {
//       const piece = rest.slice(0, CHUNK);
//       this.scene.send(new NetworkId(AUDIO_CH), piece);
//     }
//   }

//   /** wait for ACK: resolves true on ACK, false on timeout */
//   private waitAck(seq:number, expectMs:number): Promise<boolean> {
//     const ACK_MIN = 1000;
//     const ACK_MAX = 15000;
//     const timeoutMs = Math.min(ACK_MAX, Math.max(ACK_MIN, Math.floor(expectMs * 1.3) + 250));

//     return new Promise<boolean>((resolve) => {
//       const timer = setTimeout(() => {
//         this.ackWaiters.delete(seq);
//         resolve(false);
//       }, timeoutMs);
//       this.ackWaiters.set(seq, { resolve, timeout: timer });
//     });
//   }

//   private handleAck(jsonStr:string) {
//     try {
//       const msg = JSON.parse(jsonStr);
//       if (msg && msg.type === "SentenceDone" && typeof msg.seq === "number") {
//         const waiter = this.ackWaiters.get(msg.seq);
//         if (waiter) {
//           clearTimeout(waiter.timeout);
//           this.ackWaiters.delete(msg.seq);
//           waiter.resolve(true);
//         }
//       }
//     } catch { /* ignore */ }
//   }

//   // ----------------- turns -----------------
//   private enqueueTurn(msg:string, targetPeer:string){
//     this.turnQueue.push({msg, targetPeer});
//     if (!this.busy) this.processTurns();
//   }

//   private async processTurns() {
//     this.busy = true;
//     while (this.turnQueue.length) {
//       const turn = this.turnQueue.shift()!;
//       this.targetPeer = turn.targetPeer;

//       const reply = await this.llmOnce(turn.msg);
//       const sentences = this.splitSentences(reply);

//       for (const sentence of sentences) {
//         const seq = ++this.seq;

//         const anim  = await this.animOnce(sentence);
//         const style = animToStyle[anim] || "neutral";
//         const pcm   = await this.ttsFramedOnce(sentence, style, 1.0);

//         this.sendHeader(seq, anim, pcm.length);
//         await new Promise(r=>setImmediate(r)); // let Unity parse header first
//         this.streamPcm(pcm);

//         const ms = Math.ceil(1000 * pcm.length / BYTES_PER_SECOND);
//         await this.waitAck(seq, ms);
//       }
//     }
//     this.busy = false;
//   }

//   definePipeline() {
//     // UI input (channel 97)
//     this.components.chatReader!.on("data", (data:any) => {
//       try {
//         const raw = data.message?.toString?.();
//         const payload = JSON.parse(raw);
//         const msg = (payload.message || "").trim();
//         const targetPeer = payload.targetPeer || "default";
//         if (!msg) return;
//         this.enqueueTurn(msg, targetPeer);
//       } catch {
//         // ignore malformed UI packets
//       }
//     });

//     // ACK listener (same channel 95)
//     this.components.ackReader!.on("data", (data:any) => {
//       const buf = data.message;
//       if (!buf || !buf.length) return;
//       if (buf[0] !== "{".charCodeAt(0)) return; // ACKs are JSON
//       this.handleAck(buf.toString());
//     });
//   }
// }

// if (fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
//   const __dirname = path.dirname(fileURLToPath(import.meta.url));
//   const absConfig = path.resolve(__dirname, "./config.json");
//   const app = new ConversationalAgent(absConfig);
//   app.start();
// }


// VERSION WITHOUT TONE
// import { NetworkId } from "ubiq";
// import { ApplicationController } from "../../components/application";
// import { TextToSpeechService } from "../../services/text_to_speech/service";
// import { TextGenerationService } from "../../services/text_generation/service";
// import { AnimationsService } from "../../services/animations/service";
// import { MessageReader } from "../../components/message_reader";
// import path from "path";
// import { fileURLToPath } from "url";

// /** Wire constants */
// const SR = 48000;
// const BYTES_PER_SECOND = 2 * SR;   // 16-bit mono
// const CHUNK = 16000;               // bytes
// const AUDIO_CH = 95;               // header + PCM + ACK live here

// type AckWaiter = { resolve:(v:boolean)=>void, timeout:NodeJS.Timeout };

// export class ConversationalAgent extends ApplicationController {
//   components: {
//     chatReader?: MessageReader;            // 97 UI input
//     ackReader?: MessageReader;             // 95 back-ACKs from Unity
//     textGenerationService?: TextGenerationService;
//     textToSpeechService?: TextToSpeechService;
//     animationsService?: AnimationsService;
//   } = {};

//   private busy = false;
//   private turnQueue: Array<{ msg: string; targetPeer: string }> = [];
//   private seq = 0;
//   private targetPeer = "default";
//   private ackWaiters: Map<number,AckWaiter> = new Map();

//   constructor(configFile: string = "config.json") { super(configFile); }

//   start(): void {
//     this.registerComponents();
//     this.joinRoom().then(() => this.definePipeline());
//   }

//   registerComponents() {
//     this.components.textGenerationService = new TextGenerationService(this.scene);
//     this.components.textToSpeechService   = new TextToSpeechService(this.scene);
//     this.components.animationsService     = new AnimationsService(this.scene);
//     this.components.chatReader            = new MessageReader(this.scene, 97);
//     this.components.ackReader             = new MessageReader(this.scene, AUDIO_CH);
//   }

//   // ----------------- helpers -----------------
//   private splitSentences(text: string): string[] {
//     const cleaned = (text || "").replace(/\s+/g, " ").trim();
//     if (!cleaned) return [];
//     const parts = cleaned.match(/[^.!?]+[.!?]*/g) || [];
//     return parts.map(s => s.trim()).filter(Boolean);
//   }

//   private onceFromService(
//     svc: any,
//     _label: string,
//     send: () => void,
//     timeoutMs = 20000
//   ): Promise<Buffer> {
//     return new Promise((resolve, reject) => {
//       let settled = false;
//       const finish = (ok: boolean, payload?: Buffer | any) => {
//         if (settled) return;
//         settled = true;
//         clearTimeout(to);

//         const off = (ev:string, fn:any)=>{ svc.off?.(ev, fn); svc.removeListener?.(ev, fn); };
//         off("response", onResponse);
//         off("data", onData);
//         off("error", onErr);

//         if (ok) {
//           resolve(Buffer.isBuffer(payload) ? payload : Buffer.from(payload ?? ""));
//         } else {
//           reject(new Error("service error"));
//         }
//       };
//       const onResponse = (b: Buffer) => finish(true, b);
//       const onData     = (b: Buffer) => finish(true, b);
//       const onErr      = (_e: any)    => finish(false);
//       const to = setTimeout(() => finish(false), timeoutMs);

//       svc.on?.("response", onResponse);
//       svc.on?.("data", onData);
//       svc.on?.("error", onErr);

//       send();
//     });
//   }

//   /** TTS once: expects "LEN:<n>\n" then n PCM bytes on "data" */
//   private ttsFramedOnce(sentence: string, timeoutMs=25000): Promise<Buffer> {
//     return new Promise((resolve, reject) => {
//       const tts:any = this.components.textToSpeechService!;
//       let mode:"header"|"body" = "header";
//       let header = Buffer.alloc(0), body = Buffer.alloc(0);
//       let remaining = 0;

//       const onData = (raw:Buffer) => {
//         let data = Buffer.isBuffer(raw) ? raw : Buffer.from(raw as any);
//         while (data.length) {
//           if (mode === "header") {
//             header = Buffer.concat([header, data]);
//             const nl = header.indexOf(0x0A);
//             if (nl === -1) return;
//             const line = header.slice(0,nl).toString("utf8").trim();
//             data = header.slice(nl+1);
//             header = Buffer.alloc(0);
//             if (!line.startsWith("LEN:")) continue;
//             remaining = parseInt(line.slice(4).trim(),10) || 0;
//             body = Buffer.alloc(0); mode = "body";
//           } else {
//             const take = Math.min(remaining, data.length);
//             if (take) {
//               body = Buffer.concat([body, data.slice(0,take)]);
//               remaining -= take; data = data.slice(take);
//             }
//             if (remaining === 0) {
//               cleanup();
//               resolve(body);
//               return;
//             }
//           }
//         }
//       };
//       const onErr = (_e:any)=>{ cleanup(); reject(new Error("tts error")); };
//       const to = setTimeout(()=>{ cleanup(); reject(new Error("tts timeout")); }, timeoutMs);
//       const cleanup = ()=> {
//         clearTimeout(to);
//         tts.off?.("data", onData); tts.off?.("error", onErr);
//         tts.removeListener?.("data", onData); tts.removeListener?.("error", onErr);
//       };

//       tts.on?.("data", onData);
//       tts.on?.("error", onErr);
//       tts.sendToChildProcess("default", sentence + "\n");
//     });
//   }

//   private async llmOnce(prompt: string): Promise<string> {
//     const svc: any = this.components.textGenerationService!;
//     const buf = await this.onceFromService(svc, "LLM", () =>
//       svc.sendToChildProcess("default", prompt + "\n"),
//       30000
//     );
//     let s = buf.toString();
//     if (s.startsWith(">")) s = s.slice(1);
//     return s.replace(/(\r\n|\n|\r)/g, " ").trim();
//   }

//   private async animOnce(sentence: string): Promise<string> {
//     const svc: any = this.components.animationsService!;
//     const buf = await this.onceFromService(svc, "ANIM", () =>
//       svc.sendToChildProcess("default", sentence + "\n"),
//       15000
//     );
//     const name = (Buffer.isBuffer(buf) ? buf.toString() : String(buf)).trim();
//     return name || "Talking";
//   }

//   private sendHeader(seq:number, anim:string, pcmLen:number) {
//     const header = {
//       type: "A",
//       seq,
//       targetPeer: this.targetPeer,
//       audioLength: String(pcmLen),
//       animationTitle: anim || "Talking",
//       sampleRate: SR,
//     };
//     this.scene.send(new NetworkId(AUDIO_CH), header);
//   }

//   private streamPcm(pcm:Buffer) {
//     for (let rest = pcm; rest.length; rest = rest.slice(CHUNK)) {
//       const piece = rest.slice(0, CHUNK);
//       this.scene.send(new NetworkId(AUDIO_CH), piece);
//     }
//   }

//   /** wait for ACK: resolves true on ACK, false on timeout */
//   private waitAck(seq:number, expectMs:number): Promise<boolean> {
//     const ACK_MIN = 1000;
//     const ACK_MAX = 15000;
//     const timeoutMs = Math.min(ACK_MAX, Math.max(ACK_MIN, Math.floor(expectMs * 1.3) + 250));

//     return new Promise<boolean>((resolve) => {
//       const timer = setTimeout(() => {
//         this.ackWaiters.delete(seq);
//         resolve(false);
//       }, timeoutMs);
//       this.ackWaiters.set(seq, { resolve, timeout: timer });
//     });
//   }

//   private handleAck(jsonStr:string) {
//     try {
//       const msg = JSON.parse(jsonStr);
//       if (msg && msg.type === "SentenceDone" && typeof msg.seq === "number") {
//         const waiter = this.ackWaiters.get(msg.seq);
//         if (waiter) {
//           clearTimeout(waiter.timeout);
//           this.ackWaiters.delete(msg.seq);
//           waiter.resolve(true);
//         }
//       }
//     } catch { /* ignore */ }
//   }

//   // ----------------- turns -----------------
//   private enqueueTurn(msg:string, targetPeer:string){
//     this.turnQueue.push({msg, targetPeer});
//     if (!this.busy) this.processTurns();
//   }

//   private async processTurns() {
//     this.busy = true;
//     while (this.turnQueue.length) {
//       const turn = this.turnQueue.shift()!;
//       this.targetPeer = turn.targetPeer;

//       const reply = await this.llmOnce(turn.msg);
//       const sentences = this.splitSentences(reply);

//       for (const sentence of sentences) {
//         const seq = ++this.seq;

//         const anim = await this.animOnce(sentence);
//         const pcm  = await this.ttsFramedOnce(sentence);

//         this.sendHeader(seq, anim, pcm.length);
//         await new Promise(r=>setImmediate(r)); // let Unity parse header first
//         this.streamPcm(pcm);

//         const ms = Math.ceil(1000 * pcm.length / BYTES_PER_SECOND);
//         await this.waitAck(seq, ms);
//       }
//     }
//     this.busy = false;
//   }

//   definePipeline() {
//     // UI input (channel 97)
//     this.components.chatReader!.on("data", (data:any) => {
//       try {
//         const raw = data.message?.toString?.();
//         const payload = JSON.parse(raw);
//         const msg = (payload.message || "").trim();
//         const targetPeer = payload.targetPeer || "default";
//         if (!msg) return;
//         this.enqueueTurn(msg, targetPeer);
//       } catch {
//         // ignore malformed UI packets
//       }
//     });

//     // ACK listener (same channel 95)
//     this.components.ackReader!.on("data", (data:any) => {
//       const buf = data.message;
//       if (!buf || !buf.length) return;
//       if (buf[0] !== "{".charCodeAt(0)) return; // ACKs are JSON
//       this.handleAck(buf.toString());
//     });
//   }
// }

// if (fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
//   const __dirname = path.dirname(fileURLToPath(import.meta.url));
//   const absConfig = path.resolve(__dirname, "./config.json");
//   const app = new ConversationalAgent(absConfig);
//   app.start();
// }