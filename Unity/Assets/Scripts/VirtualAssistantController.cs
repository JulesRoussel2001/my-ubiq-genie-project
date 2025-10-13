using System.Collections;
using UnityEngine;

public class VirtualAssistantController : MonoBehaviour
{
    [Header("Animation")]
    [SerializeField] private Animator animator;
    [SerializeField] private int animatorLayer = 0;
    [SerializeField] private bool forceSafeAnimatorSettings = true;

    [Header("Idle")]
    [Tooltip("State to return to when speech ends (on ACK from CAM).")]
    [SerializeField] private string idleState = "Idle";

    [Header("Timing")]
    [Tooltip("Fallback clip length (s) if runtime length can't be read yet.")]
    [SerializeField] private float defaultClipSeconds = 1f;
    [Tooltip("Retrigger near the end while speech has > one clip remaining (0.90–0.98 typical).")]
    [Range(0.5f, 0.99f)] [SerializeField] private float retriggerAtNormalized = 0.95f;

    // Runtime
    private float remainingSpeechSeconds = 0f;
    private string currentStateName = null;
    private int currentStateHash = 0;
    private float currentClipSeconds = 1f; // seconds at speed = 1
    private int idleHash;

    private void Awake()
    {
        if (!animator) animator = GetComponent<Animator>();
        if (!animator) { enabled = false; return; }

        if (forceSafeAnimatorSettings)
        {
            animator.updateMode = AnimatorUpdateMode.Normal;
            animator.cullingMode = AnimatorCullingMode.AlwaysAnimate;
            animator.applyRootMotion = false;
        }

        idleHash = Animator.StringToHash(idleState);
        currentClipSeconds = defaultClipSeconds;
    }

    /// Cross-fade to any state; keep native timing.
    public void PlayAnimation(string state)
    {
        if (!enabled || animator == null || string.IsNullOrEmpty(state)) return;

        int hash = Animator.StringToHash(state);
        if (!animator.HasState(animatorLayer, hash)) return;

        currentStateName = state;
        currentStateHash = hash;

        animator.speed = 1f; // ensure native timing
        animator.CrossFade(hash, 0.05f, animatorLayer, 0f);

        // Capture active clip length next frame (robust for blend trees)
        StartCoroutine(CaptureClipLengthNextFrame());
    }

    private IEnumerator CaptureClipLengthNextFrame()
    {
        yield return null;
        currentClipSeconds = GetActiveClipLengthSeconds();
        if (currentClipSeconds <= 0f) currentClipSeconds = defaultClipSeconds;
    }

    /// Called continuously by CAM while audio is being injected.
    public void SetRemainingSpeechSeconds(float seconds)
    {
        remainingSpeechSeconds = Mathf.Max(0f, seconds);
    }

    /// Called by CAM when speech finishes (ACK). Return to Idle.
    public void OnSpeechEnded()
    {
        remainingSpeechSeconds = 0f;
        animator.speed = 1f;

        if (!string.IsNullOrEmpty(idleState) && animator.HasState(animatorLayer, idleHash))
        {
            animator.CrossFade(idleHash, 0.08f, animatorLayer, 0f);
        }
    }

    private void Update()
    {
        if (remainingSpeechSeconds <= 0f || currentStateHash == 0) return;

        var info = animator.GetCurrentAnimatorStateInfo(animatorLayer);
        bool inCurrent = (info.shortNameHash == currentStateHash) || info.IsName(currentStateName);
        if (!inCurrent) return;

        // Active clip length (prefer live value each frame)
        float clipSec = GetActiveClipLengthSeconds();
        if (clipSec <= 0f) clipSec = currentClipSeconds > 0f ? currentClipSeconds : defaultClipSeconds;

        // Wrap normalized time to [0,1) so this works for looping and non-looping clips
        float t01 = info.normalizedTime % 1f; if (t01 < 0f) t01 += 1f;

        // RULE: While we have MORE than one full clip of speech left, retrigger at the end (no speed scaling).
        if (remainingSpeechSeconds > clipSec)
        {
            if (t01 >= retriggerAtNormalized)
            {
                animator.speed = 1f; // keep native speed
                animator.Play(currentStateHash, animatorLayer, 0f);
            }
            return; // keep repeating until we drop to ≤ one clip
        }

        // Once remainingSpeechSeconds ≤ clip length:
        // - Do NOT retrigger
        // - Do NOT change speed
        // Let the current play finish naturally; CAM will call OnSpeechEnded() → Idle.
    }

    /// Length (seconds) of the primary active clip on the current state/layer.
    private float GetActiveClipLengthSeconds()
    {
        var clips = animator.GetCurrentAnimatorClipInfo(animatorLayer);
        if (clips != null && clips.Length > 0 && clips[0].clip)
            return Mathf.Max(0f, clips[0].clip.length);
        return 0f;
    }
}