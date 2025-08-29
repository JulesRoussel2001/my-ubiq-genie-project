using System;
using System.Collections;
using UnityEngine;
using TMPro;
using UnityEngine.UI;
using Ubiq.Messaging;
using Ubiq.Networking;

[Serializable]
public class ChatMessageInput
{
    public string type;
    public string message;
}

public class ChatUIManager : MonoBehaviour
{
    [Header("UI")]
    public TMP_InputField chatInput;
    public Button         sendButton;
    public RectTransform  scrollContent;
    public GameObject     textPrefab;   // must contain a TMP_Text child

    private readonly NetworkId CHAT_CHANNEL = new NetworkId(97);

    private NetworkContext context;

    [Serializable]
    private struct ChatMessage
    {
        public string type;
        public string targetPeer;
        public string message;
    }

    void Awake()
    {
        // keep across scene loads
        DontDestroyOnLoad(gameObject);
    }

    IEnumerator Start()
    {
        // wait for Ubiq’s persistent NetworkScene
        yield return null;

        var netSceneComp = FindObjectOfType<NetworkScene>();
        if (netSceneComp == null)
        {
            Debug.LogError("[ChatUI] Could not find NetworkScene!");
            yield break;
        }

        // reparent under that scene
        transform.SetParent(netSceneComp.transform, false);

        Debug.Log($"[ChatUI] Registering on channel {CHAT_CHANNEL}");
        context = NetworkScene.Register(this, CHAT_CHANNEL);

        sendButton.onClick.AddListener(OnSendClicked);
    }

    private void OnSendClicked()
    {
        var txt = chatInput.text.Trim();
        if (string.IsNullOrEmpty(txt)) return;

        Debug.Log($"[ChatUI] OnSendClicked: '{txt}'");

        var payload = new ChatMessageInput
        {
            type    = "ChatMessageInput",
            message = txt
        };

        var json = JsonUtility.ToJson(payload);
        Debug.Log($"[ChatUI] Sending → {json}");

        context.SendJson(payload);
        chatInput.text = "";
    }

    public void ProcessMessage(ReferenceCountedSceneGraphMessage raw)
    {
        var msg = raw.FromJson<ChatMessage>();
        if (msg.type != "ChatMessage") return;

        Debug.Log($"[ChatUI] ProcessMessage → {msg.message}");

        var go  = Instantiate(textPrefab, scrollContent);
        var tmp = go.GetComponentInChildren<TMP_Text>();
        tmp.text = $"{msg.message}";
    }
}