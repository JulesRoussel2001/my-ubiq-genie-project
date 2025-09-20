using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using Ubiq.Messaging;
using Ubiq.Networking;

/// <summary>
/// Receives audio packets on a Ubiq channel, injects them into InjectableAudioSource,
/// and asks VirtualAssistantController to play an animation on the first PCM of each sentence.
/// </summary>
public class ConversationalAgentManager : MonoBehaviour
{
    // Singleton guard (prevents double audio)
    private static ConversationalAgentManager s_active;
    private void Awake()
    {
        if (s_active != null && s_active != this)
        {
            Destroy(gameObject);
            return;
        }
        s_active = this;
    }
    private void OnDestroy() { if (s_active == this) s_active = null; }

    [Header("Ubiq")]
    [SerializeField] private NetworkId networkId = new NetworkId(95);
    private NetworkContext context;

    [Header("References")]
    public InjectableAudioSource audioSource;                 // required
    public VirtualAssistantController assistantController;    // optional

    // Sentence framing
    private bool receivingPcm = false;
    private int bytesRemaining = 0;
    private int currentSeq = -1;
    private int currentSampleRate = 48000;

    // ACK scheduling
    private int pendingAckSeq = -1;
    private float pendingAckWhen = 0f;

    // Anim timing
    private string pendingAnimTitle = null;
    private bool animStartedForSentence = false;

    // Queues & guards
    private readonly Queue<AudioHeader> headerQueue = new Queue<AudioHeader>();
    private readonly Queue<byte[]> pcmQueue = new Queue<byte[]>();
    private readonly HashSet<int> seenHeaderSeqs = new HashSet<int>();
    private int lastAckedSeq = -1;

    [Serializable]
    private struct AudioHeader
    {
        public string type;           // "A"
        public string targetPeer;     // ignored
        public string audioLength;    // bytes as string
        public string animationTitle; // e.g., "Talking"
        public int seq;               // sentence index
        public int sampleRate;        // Hz
    }

    [Serializable]
    private struct SentenceAck
    {
        public string type; // "SentenceDone"
        public int seq;
    }

    private void Start()
    {
        if (s_active != this) return;

        context = NetworkScene.Register(this, networkId);

        if (!audioSource)
        {
            Debug.LogError("[CAM] Missing InjectableAudioSource reference.");
        }
    }

    private void Update()
    {
        if (s_active != this) return;

        // Fire delayed ACK at audible end
        if (pendingAckSeq != -1 && Time.time >= pendingAckWhen)
        {
            SendAck(pendingAckSeq);
            pendingAckSeq = -1;
            DrainQueuesIfPossible();
        }
    }

    // Receiver for channel 95 (audio)
    public void ProcessMessage(ReferenceCountedSceneGraphMessage data)
    {
        if (s_active != this) return;
        if (!audioSource) return;

        var arr = data.data.ToArray();
        if (arr == null || arr.Length == 0) return;

        // Header?
        if (arr[0] == (byte)'{')
        {
            try
            {
                var json = Encoding.UTF8.GetString(arr);
                var header = JsonUtility.FromJson<AudioHeader>(json);

                if (seenHeaderSeqs.Contains(header.seq)) return;
                seenHeaderSeqs.Add(header.seq);

                headerQueue.Enqueue(header);
                DrainQueuesIfPossible();
                return;
            }
            catch
            {
                // fall through as PCM if JSON parse failed
            }
        }

        // PCM chunk
        pcmQueue.Enqueue(arr);
        DrainQueuesIfPossible();
    }

    private void DrainQueuesIfPossible()
    {
        if (s_active != this) return;

        if (!receivingPcm && headerQueue.Count > 0 && pendingAckSeq == -1)
        {
            var header = headerQueue.Dequeue();

            if (!int.TryParse(header.audioLength, out bytesRemaining)) bytesRemaining = 0;
            currentSeq        = header.seq;
            currentSampleRate = header.sampleRate > 0 ? header.sampleRate : 48000;
            receivingPcm      = bytesRemaining > 0;

            pendingAnimTitle = string.IsNullOrEmpty(header.animationTitle) ? "Talking" : header.animationTitle;
            animStartedForSentence = false;

            if (!receivingPcm)
            {
                SendAck(currentSeq);
                return;
            }
        }

        while (receivingPcm && pcmQueue.Count > 0 && bytesRemaining > 0)
        {
            var chunk = pcmQueue.Dequeue();
            int take = Mathf.Min(bytesRemaining, chunk.Length);
            if (take <= 0) continue;

            if (!animStartedForSentence)
            {
                animStartedForSentence = true;

                if (assistantController)
                {
                    assistantController.PlayAnimation(pendingAnimTitle);
                }

                pendingAckSeq  = currentSeq;   // arm ACK now
                pendingAckWhen = Time.time;
            }

            // estimate audio time to schedule ACK at audible end
            float chunkSeconds = take / (2f * currentSampleRate);
            pendingAckWhen = Mathf.Max(pendingAckWhen, Time.time) + chunkSeconds;

            // Inject exactly 'take' bytes; re-queue leftover for next sentence
            if (take == chunk.Length)
            {
                audioSource.InjectPcm(chunk);
            }
            else
            {
                var head = new byte[take];
                Buffer.BlockCopy(chunk, 0, head, 0, take);
                audioSource.InjectPcm(head);

                var tailLen = chunk.Length - take;
                if (tailLen > 0)
                {
                    var tail = new byte[tailLen];
                    Buffer.BlockCopy(chunk, take, tail, 0, tailLen);
                    var tmpQ = new Queue<byte[]>();
                    tmpQ.Enqueue(tail);
                    while (pcmQueue.Count > 0) tmpQ.Enqueue(pcmQueue.Dequeue());
                    while (tmpQ.Count > 0) pcmQueue.Enqueue(tmpQ.Dequeue());
                }
            }

            bytesRemaining -= take;

            if (bytesRemaining <= 0)
            {
                receivingPcm = false;
                return;
            }
        }
    }

    private void SendAck(int seq)
    {
        if (seq == lastAckedSeq) return;
        lastAckedSeq = seq;

        var ack = new SentenceAck { type = "SentenceDone", seq = seq };
        var json = JsonUtility.ToJson(ack);
        context.Send(json);
    }
}
