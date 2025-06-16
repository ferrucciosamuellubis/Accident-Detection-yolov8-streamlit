import streamlit as st
from pathlib import Path
from PIL import Image
from utils.helper import Helper
from utils.settings import Settings

class AccidentDetectionApp:
    def __init__(self):
        """
        Inisialisasi awal aplikasi.
        Mengatur konfigurasi halaman dan inisialisasi helper & settings.
        """
        st.set_page_config(
            page_title="Deteksi Kecelakaan | YOLOv8",
            page_icon="💥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self.settings = Settings()
        self.helper = Helper()
        self.model = None  # Inisialisasi model sebagai None

    def load_model(self):
        """
        Memuat model YOLO menggunakan helper.
        """
        model_name, model_path = self.settings.available_models[0]
        return self.helper.load_model(model_name, Path(model_path))

    def show_detection_page(self):
        """
        Menampilkan konten halaman deteksi utama.
        """
        st.sidebar.header("Pengaturan Model")
        self.confidence = float(st.sidebar.slider(
            "Pilih Confidence Model", 25, 100, 40)) / 100
        
        source_radio = st.sidebar.radio(
            "Pilih Sumber",
            self.settings.sources_list,
        )

        if source_radio == self.settings.IMAGE:
            self.upload_and_detect_image()
        elif source_radio == self.settings.VIDEO:
            self.helper.play_stored_video(self.confidence, self.model)
        else:
            st.error("Sumber tidak valid, silakan pilih lagi.")

    def upload_and_detect_image(self):
        """
        Menangani unggah gambar dan proses deteksi.
        """
        uploaded_image = st.sidebar.file_uploader(
            "Unggah sebuah gambar", type=["jpg", "jpeg", "png"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Gambar Asli", use_container_width=True)
        
        with col2:
            if uploaded_image:
                if st.sidebar.button('Deteksi Objek'):
                    # Memanggil metode predict pada model yang sudah dimuat
                    res = self.model.predict(Image.open(uploaded_image), conf=self.confidence)
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Gambar Hasil Deteksi', use_container_width=True)
                    
                    with st.expander("Hasil Deteksi (Teks)"):
                        if len(boxes) > 0:
                            for box in boxes:
                                st.write(box.data)
                        else:
                            st.write("Tidak ada objek yang terdeteksi.")


    def run(self):
        """
        Fungsi utama untuk menjalankan aplikasi Streamlit.
        """
        st.title("Deteksi Kecelakaan menggunakan YOLOv8")
        st.markdown(
            "Aplikasi ini mendeteksi kecelakaan dari gambar atau video menggunakan model YOLOv8."
        )

        # Langkah 1: Coba muat model terlebih dahulu
        self.model = self.load_model()
        
        # Langkah 2: Periksa apakah model berhasil dimuat atau tidak
        if self.model is not None:
            # Jika model berhasil dimuat, tampilkan halaman deteksi
            self.show_detection_page()
        else:
            # Jika model gagal dimuat (helper mengembalikan None), tampilkan pesan dan hentikan aplikasi
            st.warning("Aplikasi tidak dapat berjalan karena model gagal dimuat. Silakan periksa konsol untuk detail error.")
            st.stop()


# Entry point untuk menjalankan aplikasi
if __name__ == "__main__":
    app = AccidentDetectionApp()
    app.run()
