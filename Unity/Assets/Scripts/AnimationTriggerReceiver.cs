using System;
using UnityEngine;
using Ubiq.Messaging;
using Ubiq.Networking;

public class AnimationTriggerReceiver : MonoBehaviour
{
    private NetworkContext context;
    private readonly NetworkId ANIM_ID = new NetworkId(96);

    [Serializable]
    private struct AnimationHeader
    {
        public string type;
        public string targetPeer;
        public string animationTitle;
        public int seq;
    }

    void Start()
    {
        context = NetworkScene.Register(this, ANIM_ID);
    }

    // Log-only to avoid double triggers; authoritative trigger happens in ConversationalAgentManager on 95.
    public void ProcessMessage(ReferenceCountedSceneGraphMessage raw)
    {
        try
        {
            var msg = raw.FromJson<AnimationHeader>();
            Debug.Log($"[AnimationTriggerReceiver] (log) anim '{msg.animationTitle}' seq={msg.seq}");
        }
        catch
        {
            // ignore non-JSON or extraneous data
        }
    }
}