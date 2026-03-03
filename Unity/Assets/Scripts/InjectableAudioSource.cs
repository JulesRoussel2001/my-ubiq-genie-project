using System;
using System.Collections.Concurrent;
using UnityEngine;

/// <summary>
/// Pushes 16-bit mono PCM into the AudioSource via OnAudioFilterRead.
/// Assumes source PCM is 48 kHz (matches your Azure TTS).
/// </summary>
[RequireComponent(typeof(AudioSource))]
public class InjectableAudioSource : MonoBehaviour
{
    [Header("Debug")]
    public bool verboseLogging = true;

    private readonly ConcurrentQueue<float> samples = new ConcurrentQueue<float>();
    private AudioClip clip;

    private static string Ts() => $"{Time.realtimeSinceStartup:F3}s";

    private void Start()
    {
        // 1 second buffer of ones to piggyback spatialization via *=
        int sr = AudioSettings.outputSampleRate;
        var ones = new float[sr];
        for (int i = 0; i < ones.Length; i++) ones[i] = 1f;

        clip = AudioClip.Create("Injectable", ones.Length, 1, sr, false);

        var audioSource = GetComponent<AudioSource>();
        audioSource.clip = clip;
        audioSource.loop = true;              // keep base signal alive
        clip.SetData(ones, 0);
        audioSource.Play();

        if (verboseLogging) Debug.Log($"[{Ts()}][Injectable] init sr={sr}, ones={ones.Length}");
    }

    private void OnDestroy()
    {
        if (clip != null)
        {
            Destroy(clip);
            clip = null;
        }
    }

    /// <summary>
    /// Enqueue 48 kHz mono 16-bit PCM (little endian).
    /// </summary>
    public void InjectPcm(Span<byte> bytes)
    {
        int sampleCount = bytes.Length / 2;
        for (int i = 0; i < sampleCount; i++)
        {
            float sample = (short)(bytes[i * 2] | (bytes[i * 2 + 1] << 8)) / 32768f;
            samples.Enqueue(sample);
        }
        if (verboseLogging)
        {
            Debug.Log($"[{Ts()}][Injectable] enqueue {bytes.Length} bytes ({sampleCount} samples)");
        }
    }

    private void OnAudioFilterRead(float[] data, int channels)
    {
        for (int dataIdx = 0; dataIdx < data.Length; dataIdx += channels)
        {
            float s = 0f;
            samples.TryDequeue(out s);

            for (int ch = 0; ch < channels; ch++)
            {
                data[dataIdx + ch] *= s;
            }
        }
    }
}