using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Ubiq.Networking;
using Ubiq.Dictionaries;
using Ubiq.Messaging;
using Ubiq.Logging.Utf8Json;
using Ubiq.Rooms;
using Ubiq.Voip;
using Ubiq.Samples;
using Ubiq.Avatars;

public class VirtualAssistantController : MonoBehaviour
{
    [Header("Animation / Movement")]
    public HandMover handMover;
    public float turnSpeed = 10.0f;
    public Animator animator;

    private string assistantSpeechTargetPeerName;
    private float assistantSpeechVolume;
    private IPeer lastTargetPeer;

    private RoomClient roomClient;
    private AvatarManager avatarManager;

    private const float SPEECH_VOLUME_FLOOR = 0.005f;

    // Called by ConversationalAgentManager to drive hand animations
    public void UpdateAssistantSpeechStatus(string targetPeerName, float volume)
    {
        assistantSpeechTargetPeerName = targetPeerName;
        assistantSpeechVolume = volume;
    }

    void Update()
    {
        UpdateHands();
        UpdateTurn();
    }

    private void UpdateHands()
    {
        if (handMover == null) return;

        if (assistantSpeechVolume > SPEECH_VOLUME_FLOOR)
        {
            handMover.Play();
        }
        else
        {
            handMover.Stop();
        }
    }

    // This is the method AnimationTriggerReceiver calls
    public void PlayAnimation(string animName)
    {
        if (animator == null) return;

        // The Animator must have states or triggers matching these names
        animator.Play(animName);
    }

    private void UpdateTurn()
    {
        // Lazy‐initialize RoomClient and AvatarManager
        if (roomClient == null)
        {
            roomClient = NetworkScene.Find(this).GetComponent<RoomClient>();
            if (roomClient == null) return;
        }
        if (avatarManager == null)
        {
            avatarManager = roomClient.GetComponentInChildren<AvatarManager>();
            if (avatarManager == null) return;
        }

        IPeer targetPeer = null;

        // 1) If LLM told us who to address, find them
        if (!string.IsNullOrEmpty(assistantSpeechTargetPeerName))
        {
            foreach (var peer in roomClient.Peers)
            {
                if (peer["ubiq.samples.social.name"] == assistantSpeechTargetPeerName)
                {
                    targetPeer = peer;
                    break;
                }
            }
            if (roomClient.Me["ubiq.samples.social.name"] == assistantSpeechTargetPeerName)
            {
                targetPeer = roomClient.Me;
            }
        }
        // 2) Otherwise, pick whoever is currently speaking the loudest
        else
        {
            float loudest = 0f;
            foreach (var avatar in avatarManager.Avatars)
            {
                var src = avatar.GetComponentInChildren<AudioSource>();
                if (src == null) continue;

                var vol = src.GetComponent<AudioSourceVolume>()?.volume ?? 0f;
                if (vol > loudest && vol > SPEECH_VOLUME_FLOOR)
                {
                    loudest = vol;
                    targetPeer = avatar.Peer;
                }
            }
        }

        // Fallback to last known
        if (targetPeer == null)
        {
            targetPeer = lastTargetPeer;
            if (targetPeer == null) return;
        }

        // Find that peer’s avatar to get head position
        Ubiq.Avatars.Avatar targetAvatar = null;
        foreach (var avatar in avatarManager.Avatars)
        {
            if (avatar.Peer == targetPeer)
            {
                targetAvatar = avatar;
                break;
            }
        }
        if (targetAvatar == null) return;

        var floating = targetAvatar.GetComponentInChildren<FloatingAvatar>();
        if (floating == null) return;

        // Turn to face them
        var pos = floating.head.position;
        var dir = new Vector3(pos.x - transform.position.x, 0, pos.z - transform.position.z);
        transform.rotation = Quaternion.Slerp(
            transform.rotation,
            Quaternion.LookRotation(dir),
            turnSpeed * Time.deltaTime
        );

        lastTargetPeer = targetPeer;
    }
}