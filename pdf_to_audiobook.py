import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import re
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/output/conversion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TTSEngine:
    """TTS Engine wrapper with fallback support"""

    def __init__(self, language='es'):
        self.language = language
        self.kokoro_available = False
        self.gtts_available = False
        self.setup_engines()

    def setup_engines(self):
        """Setup available TTS engines"""
        # Try Kokoro first
        try:
            from kokoro import KPipeline
            self.kokoro_pipeline = KPipeline(lang_code=self.language)
            self.kokoro_available = True
            logger.info("✅ Kokoro TTS engine initialized")
        except Exception as e:
            logger.warning(f"❌ Kokoro not available: {e}")
            self.kokoro_available = False

        # Try gTTS as fallback
        try:
            from gtts import gTTS
            import tempfile
            self.gtts_available = True
            logger.info("✅ gTTS engine initialized")
        except Exception as e:
            logger.warning(f"❌ gTTS not available: {e}")
            self.gtts_available = False

        if not self.kokoro_available and not self.gtts_available:
            raise Exception("No TTS engines available!")

    def generate_audio(self, text, voice='ef_dora', speed=1, split_pattern=r'\n+'):
        """Generate audio using available TTS engine"""
        if self.kokoro_available:
            return self._generate_kokoro(text, voice, speed, split_pattern)
        elif self.gtts_available:
            return self._generate_gtts(text)
        else:
            raise Exception("No TTS engines available!")

    def _generate_kokoro(self, text, voice, speed, split_pattern):
        """Generate audio using Kokoro"""
        try:
            generator = self.kokoro_pipeline(text, voice=voice, speed=1, split_pattern=r'\n+')
            audio_segments = []

            for seg_num, (gs, ps, audio) in enumerate(generator):
                if audio is not None and len(audio) > 0:
                    audio_segments.append(audio)

            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                return combined_audio
            else:
                raise Exception("Kokoro generated no audio")

        except Exception as e:
            logger.error(f"Kokoro failed: {e}")
            # Fallback to gTTS if available
            if self.gtts_available:
                logger.info("Falling back to gTTS...")
                return self._generate_gtts(text)
            else:
                raise

    def _generate_gtts(self, text):
        """Generate audio using gTTS"""
        try:
            from gtts import gTTS
            import tempfile
            import io
            from pydub import AudioSegment
            from pydub.generators import Sine

            # Create gTTS object
            tts = gTTS(text=text, lang=self.language, slow=False)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                tts.save(tmp_file.name)

                # Convert to WAV format with proper sample rate
                audio = AudioSegment.from_mp3(tmp_file.name)
                audio = audio.set_frame_rate(24000).set_channels(1)

                # Convert to numpy array
                samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

                # Cleanup
                os.unlink(tmp_file.name)

                return samples

        except Exception as e:
            logger.error(f"gTTS failed: {e}")
            raise

class PDFToAudiobookConverter:
    def __init__(self, input_path, output_dir, language='es', voice='af_heart'):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.language = language
        self.voice = voice
        self.output_dir.mkdir(exist_ok=True)

        # Initialize TTS engine
        logger.info("🚀 Initializing TTS engine...")
        self.tts_engine = TTSEngine(language=language)

        # Test TTS functionality
        self.test_tts_functionality()

    def test_tts_functionality(self):
        """Test TTS with sample texts"""
        logger.info("🧪 Testing TTS functionality...")

        test_texts = [
            "Hola mundo. Esta es una prueba del sistema de texto a voz.",
            "La nostalgia es un monstruo que se alimenta de saves abandonados.",
            "Bienvenidos al Limbo de los Píxeles. Elige tu propia dificultad."
        ]

        for i, text in enumerate(test_texts):
            logger.info(f"🧪 Test {i+1}: '{text}'")

            try:
                audio_data = self.tts_engine.generate_audio(text, self.voice)
                duration = len(audio_data) / 24000

                # Save test file
                test_file = self.output_dir / f"test_{i+1}.wav"
                import soundfile as sf
                sf.write(str(test_file), audio_data, 24000)

                logger.info(f"  ✅ Generated: {duration:.2f}s → {test_file}")

            except Exception as e:
                logger.error(f"  ❌ Test failed: {e}")

    def extract_text_from_pdf(self):
        """Extract text from PDF"""
        logger.info(f"📖 Extracting text from: {self.input_path}")

        if not self.input_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.input_path}")

        try:
            import PyPDF2

            with open(self.input_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                logger.info(f"Document has {num_pages} pages")

                all_text = []
                for page_num in tqdm(range(num_pages), desc="Extracting pages"):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            cleaned_text = self.clean_text(page_text)
                            all_text.append(cleaned_text)
                    except Exception as e:
                        logger.error(f"Error on page {page_num + 1}: {e}")
                        continue

                full_text = " ".join(all_text)
                logger.info(f"📄 Extracted {len(full_text)} characters")
                return full_text

        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise

    def clean_text(self, text):
        """Clean extracted text"""
        if not text:
            return ""

        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r'([.!?])\s*', r'\1 ', text)  # Fix punctuation spacing
        return text.strip()


    def split_text_into_chunks(self, text, max_chars=1000):
        """Split text into manageable chunks"""
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence would make the chunk too long, start a new chunk
            if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        logger.info(f"📝 Split into {len(chunks)} chunks")
        return chunks

    def convert(self):
        """Main conversion method"""
        logger.info("🚀 Starting conversion...")

        try:
            # Extract text
            text = self.extract_text_from_pdf()
            if not text:
                logger.error("No text extracted")
                return False

            # Split into chunks
            chunks = self.split_text_into_chunks(text)

            # Generate audio for each chunk
            all_audio = []
            successful_chunks = 0

            for i, chunk in enumerate(tqdm(chunks, desc="Generating audio")):
                if not chunk.strip():
                    continue

                try:
                    logger.info(f"🎵 Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
                    audio_data = self.tts_engine.generate_audio(chunk, self.voice)

                    # Save individual chunk for debugging
                    chunk_file = self.output_dir / f"chunk_{i+1:04d}.wav"
                    import soundfile as sf
                    sf.write(str(chunk_file), audio_data, 24000)

                    all_audio.append(audio_data)
                    successful_chunks += 1

                    duration = len(audio_data) / 24000
                    logger.info(f"  ✅ Chunk {i+1}: {duration:.2f}s")

                except Exception as e:
                    logger.error(f"  ❌ Chunk {i+1} failed: {e}")
                    continue

            logger.info(f"📊 Generated {successful_chunks}/{len(chunks)} chunks successfully")

            if not all_audio:
                logger.error("❌ No audio generated")
                return False

            # Combine all audio
            logger.info("🔗 Combining audio chunks...")
            combined_audio = np.concatenate(all_audio)
            output_file = self.output_dir / "audiobook.wav"
            import soundfile as sf
            sf.write(str(output_file), combined_audio, 24000)

            total_duration = len(combined_audio) / 24000
            logger.info(f"🎉 Audiobook created: {output_file}")
            logger.info(f"⏱️ Duration: {total_duration:.2f}s ({total_duration/60:.2f} min)")

            return True

        except Exception as e:
            logger.error(f"💥 Conversion failed: {e}")
            return False

def main():
    # Configuration
    INPUT_PDF = "/app/input/Disco22.pdf"
    OUTPUT_DIR = "/app/output"

    # Check if input exists
    if not os.path.exists(INPUT_PDF):
        logger.error(f"❌ Input PDF not found: {INPUT_PDF}")
        logger.info("💡 Please place your PDF in the 'input' directory")
        return

    try:
        # Initialize converter
        converter = PDFToAudiobookConverter(
            input_path=INPUT_PDF,
            output_dir=OUTPUT_DIR,
            language='e',  # Spanish
            voice='ef_dora'
        )

        # Perform conversion
        success = converter.convert()

        if success:
            logger.info("✅ Conversion completed successfully!")
        else:
            logger.error("❌ Conversion failed!")

    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")

if __name__ == "__main__":
    main()