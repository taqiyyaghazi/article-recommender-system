# Laporan Proyek Machine Learning - Ghazi Taqiyya Al Anshari

## Project Overview

Proyek ini berfokus pada pengembangan sistem rekomendasi berbasis konten untuk artikel, menggunakan teknik pembelajaran mesin. Sistem ini penting karena membantu pengguna menemukan artikel yang relevan dengan minat mereka, meningkatkan pengalaman membaca dan keterlibatan pengguna di platform.

Di era digital, jumlah artikel yang tersedia secara online terus meningkat, membuat pengguna kesulitan menemukan konten yang sesuai dengan minat mereka. Tanpa sistem rekomendasi yang efektif, pengguna mungkin merasa kewalahan oleh banyaknya pilihan atau melewatkan konten yang relevan. Hal ini tidak hanya menurunkan kualitas pengalaman pengguna, tetapi juga dapat mengurangi keterlibatan dan retensi di platform. Selain itu, penyedia konten menghadapi tantangan untuk memastikan bahwa artikel mereka menjangkau audiens yang tepat tanpa harus mengandalkan strategi promosi yang intensif.

Oleh karena itu, dibutuhkan sebuah sistem rekomendasi yang dapat secara otomatis menyarankan artikel berdasarkan minat pengguna, memaksimalkan relevansi dan meningkatkan keterlibatan. Proyek ini menggunakan model FastText untuk representasi teks dan cosine similarity untuk menghitung kesamaan antar artikel, memungkinkan sistem untuk merekomendasikan artikel yang paling relevan dengan preferensi setiap pengguna.

## Business Understanding

### Problem Statements

Berdasarkan Project Overview didapatkan Problem Statements sebagai berikut:

- Bagaimana cara merekomendasikan artikel yang relevan kepada pengguna berdasarkan artikel yang telah mereka baca?
- Bagaimana cara meningkatkan akurasi rekomendasi artikel dengan memanfaatkan teknik pembelajaran mesin?

### Goals

Berdasarkan Problem Statements ditentukan tujuan dari proyek ini adalah sebagai berikut:

- Mengembangkan sistem rekomendasi yang dapat menyarankan artikel-artikel yang relevan berdasarkan konten artikel yang telah dibaca pengguna.
- Mengimplementasikan teknik pembelajaran mesin untuk meningkatkan akurasi dan relevansi rekomendasi artikel.

## Data Understanding

Dataset yang digunakan dalam proyek ini berisi 337 artikel. Dataset ini bersumber dari dataset milik Hsankesara yaitu [Medium Articles](https://www.kaggle.com/datasets/hsankesara/medium-articles). Dataset tersebut memiliki kolom sebagai berikut:

- author: Penulis artikel
- claps: Jumlah respon positif dari pembaca
- reading_time: Waktu yang dibutuhkan untuk membaca artikel
- link: URL asli artikel
- title: Judul artikel
- text: Isi artikel

Dataset ini tidak memiliki nilai yang bernilai null. Visualisasi awal menggunakan Word Cloud menunjukkan bahwa mayoritas artikel membahas tentang Data, Model, dan Neural Network.

## Data Preparation

Sebelum melakukan modelling dengan data dilakukan tahap praproses yang bertujuan untuk menghilangkan noise pada data agar model dapat dilatih lebih optimal. Beberapa praproses yang diterapkan adalah sebagai berikut:

1. Lowercasing: Mengubah seluruh teks menjadi huruf kecil.
2. Menghapus Tanda Baca: Menggunakan ekspresi reguler untuk menghapus semua karakter yang bukan huruf atau spasi.
3. Tokenisasi dan Stop Words Removal: Memecah teks menjadi daftar kata-kata dan menghapus kata-kata umum yang tidak penting.
4. Lemmatization: Mengubah setiap kata ke bentuk dasarnya.

Setelah itu hasil artikel yang telah di praproses di himpun dalam satu list yang kemudian akan digunakan untuk pelatihan model FastText.

### Vektorisasi dengan FastText
Proyek ini menggunakan model FastText untuk menghasilkan representasi vektor dari teks artikel. FastText adalah varian dari Word2Vec yang memperhitungkan n-grams dari kata-kata dalam proses pembelajaran, sehingga dapat menangkap informasi morfologis lebih baik daripada Word2Vec.

### Pelatihan Model FastText
Model FastText dilatih dengan parameter utama sebagai berikut:

- vector_size=100: Ukuran vektor embedding yang dihasilkan adalah 100 dimensi. Vektor ini merepresentasikan setiap kata dalam ruang vektor.
- window=5: Ukuran jendela konteks adalah 5 kata di sekitar kata target. Ini berarti model mempertimbangkan hingga 5 kata sebelum dan setelah kata target untuk memprediksi konteks.
- min_count=1: Semua kata yang muncul setidaknya sekali dalam korpus akan dipertimbangkan dalam pelatihan model. Ini memungkinkan model untuk mempelajari representasi dari kata-kata langka.
- sg=1: Parameter ini menunjukkan bahwa model menggunakan algoritma Skip-Gram, di mana model memprediksi konteks (kata-kata sekitar) berdasarkan kata target.

### Cara Kerja Algoritma FastText pada Data
Algoritma Skip-Gram, yang digunakan dalam FastText, bekerja dengan cara memprediksi kata-kata konteks di sekitar kata target dalam sebuah jendela konteks. Misalnya, jika kata target adalah "learning" dalam kalimat "Deep learning is powerful," model akan mencoba memprediksi kata-kata seperti "Deep," "is," dan "powerful." Selama pelatihan, FastText belajar untuk memetakan kata-kata yang sering muncul dalam konteks yang serupa ke dalam vektor yang berdekatan di ruang vektor.
Salah satu keunggulan FastText dibandingkan model lain adalah kemampuannya untuk menangkap informasi sub-kata. Misalnya, kata "learning" dan "learner" akan memiliki vektor yang mirip karena keduanya berbagi n-grams yang sama seperti "learn" dan "ing." Ini memungkinkan model untuk lebih tanggap terhadap variasi kata dan menangani kata-kata yang tidak dikenal (OOV) dengan lebih baik.

Selain praproses pada data text digunakan untuk pelatihan model FastText. Praproses juga dilakukan dengan mengkonversi nilai claps ke dalam nilai numeric menggunakan fungsi convert_claps_to_numeric. Fungsi convert_claps_to_numeric mengubah nilai jumlah "claps" dari format string yang menggunakan satuan ribuan (K) atau juta (M) menjadi angka numerik yang sesuai. Jika string mengandung 'K', fungsi ini menghapus 'K' dan mengalikan nilai numerik dengan 1000. Jika mengandung 'M', fungsi ini menghapus 'M' dan mengalikan nilai numerik dengan 1.000.000. Jika tidak ada satuan, fungsi hanya menghapus koma dan mengonversi string ke float. Fungsi tersebut diimplementasikan pada data claps yang kemudian disimpan pada kolom claps_numeric.

## Modeling



### Tahapan Pembuatan Model

Kesamaan antar artikel kemudian dihitung menggunakan cosine similarity, yang mengukur sudut antara dua vektor dari hasil vektorisasi artikel menggunakan FastText. Dua artikel dengan sudut yang lebih kecil (cosine similarity lebih besar) dianggap lebih mirip.

Untuk menguji sistem rekomendasi yang dibangun, diberikan input artikel dengan judul "Python for Data Science: 8 Concepts You May Have Forgotten". Dalam mendapatkan rekomendasi artikel dibuat fungsi recommend_articles dengan parameter sebagai berikut: - title: Judul artikel yang ingin dijadikan sebagai input untuk rekomendasi. Artikel ini akan dibandingkan dengan artikel lain dalam dataset. - data: DataFrame yang berisi artikel-artikel yang akan digunakan untuk mencari rekomendasi. Data ini harus memiliki kolom title dan article_vector, di mana article_vector adalah representasi vektor dari artikel yang dihasilkan oleh model FastText. - model: Model FastText yang telah dilatih dan digunakan untuk menghasilkan vektor representasi artikel. - top_n: Jumlah rekomendasi artikel yang ingin dikembalikan oleh fungsi ini. Default-nya adalah 5 artikel teratas.

Cara kerja fungsi recommend_articles adalah sebagai berikut:
a. **Mengambil Vektor Artikel Input**
Fungsi pertama-tama mencari vektor representasi dari artikel yang judulnya sesuai dengan parameter title. Vektor ini disimpan dalam variabel article_vector.
b. **Menghitung Kesamaan (Cosine Similarity)**
Fungsi kemudian menghitung kesamaan (similarity) antara vektor article_vector dari artikel input dan vektor representasi semua artikel lain dalam dataset. Ini dilakukan menggunakan metode cosine_similarity, yang mengukur kesamaan sudut antara dua vektor. Hasil kesamaan ini disimpan dalam kolom baru bernama similarity di DataFrame.
c. **Mengurutkan Artikel Berdasarkan Kesamaan**
DataFrame kemudian diurutkan berdasarkan kolom similarity dalam urutan menurun (descending), sehingga artikel dengan kesamaan tertinggi berada di urutan teratas.
d. **Mengambil Top N Rekomendasi**
Fungsi mengambil top_n + 1 artikel teratas (termasuk artikel input itu sendiri) dan mengembalikan top_n artikel paling mirip selain artikel input. Hasilnya adalah DataFrame yang hanya berisi kolom title dan similarity untuk artikel-artikel yang direkomendasikan.

Berdasarkan vektor representasi artikel yang digunakan sebagai input, sistem merekomendasikan 5 artikel teratas yang paling mirip, yaitu:
a. Taking Keras to the Zoo
b. Machine Learning Exercises In Python
c. CrAIg: Using Neural Networks to learn Mario
d. Ultimate Guide to Leveraging NLP & Machine Learning
e. C# Plays Bejeweled Blitz

## Evaluation

Untuk mengevaluasi kinerja model rekomendasi artikel yang dibangun digunakan metrik evaluasi sebagai berikut:
1. Precision
Precision mengukur proporsi prediksi positif yang benar di antara semua prediksi positif yang dilakukan oleh model. Dalam konteks rekomendasi artikel, ini berarti berapa banyak dari artikel yang direkomendasikan oleh model yang benar-benar relevan.
Precision= True Positives/(True Positives+False Positives)
Precision tinggi menunjukkan bahwa model memberikan rekomendasi yang relevan dan minim kesalahan, yaitu artikel yang direkomendasikan sebagian besar memang relevan.
2. Recall
Recall mengukur proporsi prediksi positif yang benar di antara semua artikel relevan yang sebenarnya. Dalam konteks rekomendasi artikel, ini berarti berapa banyak dari artikel relevan yang berhasil ditemukan dan direkomendasikan oleh model.
Recall=True Positives/(True Positives+False Negatives)

Untuk menghitung Precision dan Recall dari hasil rekomendasi digunakan fungsi evaluate_model. Fungsi evaluate_model bekerja dengan cara berikut: pertama, ia menginisialisasi list kosong untuk menyimpan nilai precision dan recall. Kemudian, fungsi ini mengiterasi setiap artikel dalam DataFrame. Untuk setiap artikel, judul artikel diambil dan artikel-artikel relevan diidentifikasi berdasarkan threshold tertentu pada kolom claps_numeric. Setelah itu, rekomendasi artikel diperoleh menggunakan model yang diuji. Artikel relevan di antara rekomendasi dihitung sebagai true positives. Precision dihitung sebagai proporsi artikel relevan dari semua artikel yang direkomendasikan, sementara recall dihitung sebagai proporsi artikel relevan yang ditemukan dari seluruh artikel relevan. Hasil precision dan recall untuk setiap artikel disimpan dalam list. Setelah proses iterasi selesai, rata-rata precision dan recall dihitung dari semua nilai yang telah dikumpulkan.

Dari fungsi evaluate_model didapatkan hasil evaluasi model sebagai berikut:
Average Precision: 0.8350
Average Recall: 0.0307

Dari hasil evaluasi dapat disimpulkan bahwa:
1. Model saat ini efektif dalam memberikan rekomendasi yang relevan berdasarkan artikel yang telah dibaca pengguna, seperti yang tercermin dalam precision yang tinggi. Namun, model memiliki keterbatasan dalam hal cakupan, karena recall-nya rendah. Ini berarti meskipun artikel yang direkomendasikan cenderung relevan, model tidak merekomendasikan banyak artikel relevan dari keseluruhan yang tersedia.
2. Untuk meningkatkan akurasi dan cakupan rekomendasi, perlu dilakukan penyempurnaan model. Meskipun precision tinggi menunjukkan relevansi yang baik, recall yang rendah menunjukkan perlunya perbaikan dalam menemukan lebih banyak artikel relevan. Ini bisa melibatkan penyesuaian parameter model, penambahan data relevansi, atau eksplorasi teknik rekomendasi tambahan.
​
 

​
 

