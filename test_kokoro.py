from kokoro import KPipeline

def test_kokoro():
    pipeline = KPipeline(lang_code='es')

    test_texts = [
        "Hola, este es un texto de prueba corto.",
        "Este es un texto un poco más largo que debería generar audio de mayor duración.",
        "La Mansión Edison era un lugar misterioso donde los personajes de las aventuras gráficas encontraban refugio. Los píxeles se derretían en las paredes y el tiempo parecía haberse detenido para siempre."
    ]

    for i, text in enumerate(test_texts):
        print(f"\n--- Test {i+1} ---")
        print(f"Text: {text}")
        print(f"Length: {len(text)} characters")

        generator = pipeline(text, voice='af_heart')
        audio_segments = []

        for seg_num, (gs, ps, audio) in enumerate(generator):
            duration = len(audio) / 24000 if audio is not None else 0
            print(f"  Segment {seg_num}: {duration:.2f}s")
            if audio is not None:
                audio_segments.append(audio)

        if audio_segments:
            total_audio = np.concatenate(audio_segments)
            total_duration = len(total_audio) / 24000
            print(f"Total duration: {total_duration:.2f}s")
        else:
            print("No audio generated!")

if __name__ == "__main__":
    test_kokoro()