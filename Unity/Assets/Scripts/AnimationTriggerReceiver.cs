using System;
using UnityEngine;
using Ubiq.Messaging;
using Ubiq.Networking;  // for NetworkScene and NetworkContext

public class AnimationTriggerReceiver : MonoBehaviour
{
    [Tooltip("Drag your VirtualAssistantController here")]
    public VirtualAssistantController assistantController;

    private NetworkContext context;
    private readonly NetworkId ANIM_ID = new NetworkId(96);

    [Serializable]
    private struct AnimationAudio
    {
        public string type;
        public string targetPeer;
        public string audioLength;
        public string animationTitle;
    }

    void Start()
    {
        // Register to listen on channel 96 for animation triggers
        context = NetworkScene.Register(this, ANIM_ID);
    }

    // Ubiq calls this when a message arrives on ANIM_ID
    public void ProcessMessage(ReferenceCountedSceneGraphMessage raw)
    {
        var msg = raw.FromJson<AnimationAudio>();

        // (Optionally) check targetPeer here before playing
        Debug.Log($"[AnimationTriggerReceiver] Got animation '{msg.animationTitle}'");

        // Invoke the method on your assistant controller
        assistantController.PlayAnimation(msg.animationTitle);
    }
}