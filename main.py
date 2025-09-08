import cv2
import numpy as np
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import insightface
from moviepy.editor import VideoFileClip

# ===== ضع توكن البوت هنا =====
TOKEN = "ضع_توكن_البوت_هنا"

model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=-1, det_size=(640, 640))  # CPU

user_data = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 أهلاً بك!\n"
        "أرسل لي صورتين أو فيديو:\n"
        "1️⃣ الصورة/الفيديو الهدف\n"
        "2️⃣ الصورة/الفيديو للوجه الذي تريد تبديله"
    )

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id

    if update.message.photo:
        file = await update.message.photo[-1].get_file()
        ext = ".jpg"
    elif update.message.video:
        file = await update.message.video.get_file()
        ext = ".mp4"
    else:
        await update.message.reply_text("📌 فقط الصور والفيديوهات مدعومة حالياً")
        return

    file_path = f"{user_id}_{len(user_data.get(user_id, []))}{ext}"
    await file.download_to_drive(file_path)

    if user_id not in user_data:
        user_data[user_id] = []
    user_data[user_id].append(file_path)

    if len(user_data[user_id]) == 2:
        try:
            input1, input2 = user_data[user_id]

            if input1.endswith(".mp4") or input2.endswith(".mp4"):
                cap1 = cv2.VideoCapture(input1)
                cap2 = cv2.VideoCapture(input2)
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                cap1.release()
                cap2.release()
            else:
                frame1 = cv2.imread(input1)
                frame2 = cv2.imread(input2)

            faces1 = model.get(frame1)
            faces2 = model.get(frame2)

            if faces1 and faces2:
                face1 = faces1[0]
                face2 = faces2[0]

                if input1.endswith(".mp4") or input2.endswith(".mp4"):
                    clip = VideoFileClip(input1)
                    def swap_frame(frame):
                        swapped_frame = model.swap_face(frame, face1, frame2, face2)
                        return swapped_frame
                    swapped_clip = clip.fl_image(swap_frame)
                    out_path = f"{user_id}_swapped.mp4"
                    swapped_clip.write_videofile(out_path, codec="libx264")
                else:
                    swapped_img = model.swap_face(frame1, face1, frame2, face2)
                    out_path = f"{user_id}_swapped.jpg"
                    cv2.imwrite(out_path, swapped_img)

                if out_path.endswith(".jpg"):
                    await update.message.reply_photo(open(out_path, "rb"))
                else:
                    await update.message.reply_video(open(out_path, "rb"))
            else:
                await update.message.reply_text("😢 لم أستطع العثور على وجوه واضحة في الملفات!")

        except Exception as e:
            await update.message.reply_text(f"❌ حدث خطأ أثناء التبديل: {e}")

        finally:
            for f in user_data[user_id]:
                if os.path.exists(f):
                    os.remove(f)
            user_data[user_id] = []

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO | filters.VIDEO, handle_file))
    print("✅ FaceSwap Bot يعمل الآن...")
    app.run_polling()

if __name__ == "__main__":
    main()
