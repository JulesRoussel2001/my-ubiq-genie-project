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
const BYTES_PER_SECOND = 2 * SR;   // 16-bit mono
const CHUNK = 16000;               // bytes
const AUDIO_CH = 95;               // header + PCM + ACK live here

type AckWaiter = { resolve:(v:boolean)=>void, timeout:NodeJS.Timeout };

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

  /** TTS once: expects "LEN:<n>\n" then n PCM bytes on "data" */
  private ttsFramedOnce(sentence: string, timeoutMs=25000): Promise<Buffer> {
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
            if (!line.startsWith("LEN:")) continue;
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
      tts.sendToChildProcess("default", sentence + "\n");
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
    );
    const name = (Buffer.isBuffer(buf) ? buf.toString() : String(buf)).trim();
    return name || "Talking";
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

        const anim = await this.animOnce(sentence);
        const pcm  = await this.ttsFramedOnce(sentence);

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