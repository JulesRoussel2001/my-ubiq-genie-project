using System.Collections;
using UnityEngine;

public class VirtualAssistantController : MonoBehaviour
{
    [Header("Animation")]
    [SerializeField] private Animator animator;      // Mixamo body Animator (uses NPC_Controller)
    [SerializeField] private int animatorLayer = 0;  // usually 0
    [SerializeField] private bool forceSafeAnimatorSettings = true;

    private void Awake()
    {
        if (!animator) animator = GetComponent<Animator>();
        if (!animator)
        {
            // Hard error is kept so you know if something is miswired.
            Debug.LogError("[VAC] No Animator assigned.");
            enabled = false;
            return;
        }

        if (forceSafeAnimatorSettings)
        {
            animator.updateMode = AnimatorUpdateMode.Normal;
            animator.cullingMode = AnimatorCullingMode.AlwaysAnimate;
            animator.applyRootMotion = false;
        }
    }

    public void PlayAnimation(string state)
    {
        if (!enabled || animator == null || string.IsNullOrEmpty(state)) return;

        int hash = Animator.StringToHash(state);
        if (!animator.HasState(animatorLayer, hash))
        {
            // Silently ignore unknown states in presentation mode.
            return;
        }

        animator.CrossFade(hash, 0.05f, animatorLayer, 0f);
    }
}