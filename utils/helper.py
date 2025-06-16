import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

class Helper:
    def load_model(self, model_name, model_path):
        """
        Memuat model YOLO dari path yang diberikan.
        Menampilkan feedback di UI Streamlit.
        Mengembalikan objek model jika berhasil, None jika gagal.
        """
        try:
            model = YOLO(model_path)
            st.success(f"Model '{model_name}' berhasil dimuat.", icon="✅")
            return model
        except Exception as e:
            # Menampilkan pesan error yang informatif di UI jika model gagal dimuat
            st.error(f"Gagal memuat model '{model_name}'.", icon="❌")
            st.error(f"Pastikan file model ada di path: '{model_path}' dan tidak korup.")
            # Mencetak detail error ke konsol untuk debugging
            print(f"ERROR: Terjadi kesalahan saat memuat model: {e}")
            return None

    def _display_detected_frames(self, conf, model, st_frame, image):
        """
        Menampilkan frame yang terdeteksi pada UI Streamlit.
        :param conf: Nilai confidence threshold.
        :param model: Objek model YOLO.
        :param st_frame: Objek st.image Streamlit untuk ditampilkan.
        :param image: Gambar input (np.array).
        """
        # Ubah ukuran gambar ke ukuran yang valid jika perlu
        image = cv2.resize(image, (720, int(720*(9/16))))

        # Lakukan prediksi pada gambar
        res = model.predict(image, conf=conf)

        # Plot hasil deteksi pada gambar
        res_plotted = res[0].plot()
        st_frame.image(res_plotted,
                       caption='Video Deteksi',
                       channels="BGR",
                       use_column_width=True
                       )

    def play_stored_video(self, conf, model):
        """
        Memutar video yang diunggah dan melakukan deteksi objek.
        :param conf: Nilai confidence threshold.
        :param model: Objek model YOLO.
        :return: None
        """
        source_vid = st.sidebar.file_uploader(
            label="Pilih sebuah video...",
            type=("mp4", "avi", "mov")
        )

        if source_vid:
            st.video(source_vid)

        if source_vid and st.sidebar.button("Deteksi Video"):
            # Simpan sementara video yang diunggah
            temp_video_path = os.path.join("/tmp", source_vid.name)
            with open(temp_video_path, "wb") as f:
                f.write(source_vid.getbuffer())

            try:
                vid_cap = cv2.VideoCapture(temp_video_path)
                st_frame = st.empty()
                while vid_cap.isOpened():
                    success, image = vid_cap.read()
                    if success:
                        self._display_detected_frames(conf, model, st_frame, image)
                    else:
                        vid_cap.release()
                        break
            except Exception as e:
                st.sidebar.error(f"Error memproses video: {e}")
            finally:
                # Hapus file video sementara setelah selesai
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
