import torch
from transformers import pipeline
import soundfile as sf
from rvc_python import rvc
import os

# --- PART 1: สร้างเสียงร้องด้วย BARK ---
def generate_bark_singing(lyrics, output_bark_path):
    print("📢 กำลังสร้างเสียงร้องจาก Bark...")
    device = 0 if torch.cuda.is_available() else -1
    # ใช้ bark-small เพื่อความเร็ว ถ้าคอมแรงใช้ suno/bark
    synthesizer = pipeline("text-to-speech", model="suno/bark-small", device=device)
    
    # เทคนิคใส่เครื่องหมายเพื่อให้ AI ร้องเพลง
    formatted_lyrics = f"[music] ♪ {lyrics} ♪"
    
    generation_params = {
        "do_sample": True,
        "speaker_preset": "v2/en_speaker_6" 
    }
    
    speech = synthesizer(formatted_lyrics, forward_params=generation_params)
    sf.write(output_bark_path, speech["audio"], samplerate=speech["sampling_rate"])
    print(f"✅ สร้างเสียงต้นฉบับสำเร็จ: {output_bark_path}")

# --- PART 2: เปลี่ยนเสียงด้วย RVC ---
def apply_rvc(input_wav, model_pth, final_output):
    print("🎭 กำลังแปลงเสียงเป็นโมเดลของคุณด้วย RVC...")
    rvc.convert(
        model_path=model_pth,
        f0_method='rmvpe', # คุณภาพดีที่สุดสำหรับเสียงร้อง
        f0_up_key=0,       # ปรับ +12 ถ้าอยากให้เสียงสูงขึ้น (เช่น ชายไปหญิง)
        input_path=input_wav,
        output_path=final_output
    )
    print(f"✨ เสร็จสมบูรณ์! ไฟล์สุดท้าย: {final_output}")

# --- รันขั้นตอนทั้งหมด ---
if __name__ == "__main__":
    lyrics = "Twinkle, twinkle, little star, how I wonder what you are"
    temp_bark = "temp_bark_voice.wav"
    my_model = "my_model.pth" # <--- เปลี่ยนชื่อไฟล์ให้ตรงกับที่คุณโหลดจาก Drive
    final_result = "final_ai_cover.wav"

    # 1. สร้างเสียงร้อง
    generate_bark_singing(lyrics, temp_bark)

    # 2. เปลี่ยนเสียงด้วย RVC
    if os.path.exists(my_model):
        apply_rvc(temp_bark, my_model, final_result)
    else:
        print(f"❌ ไม่พบไฟล์โมเดล {my_model} กรุณาเช็คชื่อไฟล์ให้ถูกต้อง")
