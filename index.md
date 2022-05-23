---
layout: default
---
# Table of Contents
* [Chapter 1: Image Processing and Visualization (pp.224-256)](#chapter-1-image-processing-and-visualization)
* [Chapter 2: Audio Processing and Visualization](#chapter-2-audio-processing-and-visualization)

# Textbook 
* 下山 輝昌、伊藤 淳二、露木 宏志 著「Python実践 データ加工/可視化 100本ノック」(秀和システム)

# GitHub Repo
* [2022-sem1-rinko](https://github.com/kiyalab-tmu/2022-sem1-rinko)

# Chapter 1: Image Processing and Visualization 

### Q.1: Image display (ノック61)
Load an image using cv2.imread and show it. 

### Q.2: Contents of image data (ノック62)
Check out the shape of an image and pixel values in the blue channel. 

### Q.3: Image cropping (ノック63)
Crop an image between (700,300) and (1200,800). 

### Q.4: Color histogram visualization (ノック64)
Visualize the color histogram of an image using cv2.calcHist. 

### Q.5: RGB transformation (ノック65)
Change the channel order from RGB -> BGR using cv2.cvtColor. 

### Q.6: Image data type
The following script was written to make an image underexposed. 
```
img_dim = img_rgb * 0.5
plt.imshow(img_dim)
```
However, the result seems strange:

<img src="figs/wrong_dim.png" width="384">

Please figure out why the problem has occurred, and modify the script to achieve the desired result:

<img src="figs/correct_dim.png" width="384">

### Q.7: Color scrambling
1. Take a photo using your smartphone.
2. Implement color scrambling.
3. Apply color scrambling to the photo.
* Any block size can be used.

<img src="figs/color_scrambling.png" width="384">

### Q.8: Image resizing (ノック66)
* Upsample and downsample an image.
* Try various interpolation methods and compare the results.

### Q.9: Image rotation (ノック67)
* Rotate an image.
* Flip an image (both horizontal and vertical).

### Q.10: Image processing (ノック68)
* Convert a color image to a grayscale one.
* Binarize an image.
* Apply a smoothing filter to an image (use cv2.bulr).

### Q.11: Drawing line or text in image (ノック69)
* Draw a text on an image.
* Draw a rectangle on an image.

### Q.12: Image save (ノック70)
* Save an image using cv2.imwrite

### Q.13: Block scrambling
1. Take a photo using your smartphone.
2. Implement block scrambling.
3. Apply block scrambling to your photo.

<img src="figs/block_scrambling.png" width="384">

### Q.14: Fast color space transform
* Load an RGB image and transform its color space to YCbCr.
* **Processing time limitation is within 4 second.**
* Color space transform equations are
```
Y  =  0.29900 * R +  0.58700 * G +  0.11400 * B
Cb = -0.16874 * R + -0.33126 * G +  0.50000 * B
Cr =  0.50000 * R + -0.41869 * G + -0.08100 * B
```
* TIPS: np.reshape and matrix multiplication would be helpful.

# Chapter 2: Audio Processing and Visualization

* HOMEに課題を解くための雛形をアップロードしましたので`git checkout main`よりファイルをコピーしてください
* 必ず`git checkout 自分のブランチ名`より、自分のブランチに戻ってください
* 発表では、**音声がzoomで聞けるように工夫をお願いします**

### Q.1 音データを再生してみよう & 音データを読み込んでみよう（ノック71&72）
* librosaを使用して"音声.mp3"と"携帯電話着信音.mp3"を読み込んでください（ノック72）
* "音声.mp3"と"携帯電話着信音.mp3"を再生してください（ノック71）
* vscode+jupyter環境では音源を再生できないので注意してください（ノック71）
* 音源ファイルのサンプリングレートを確認してください（ノック72）
* ロードした音源を配列に格納し、shape/max/minを確認してください（ノック72）

### Q.2 音データの一部を取得してみよう（ノック73）
* "音声.mp3"と"携帯電話着信音.mp3"を1秒のデータに変換してください
* 編集したファイルを再生してみましょう

### Q.3 音データのサンプリングレートを変えてみよう（ノック74）
* サンプリングレートとは何かを説明してください
* サンプリングレートを22050Hzにして"音声.mp3"と"携帯電話着信音.mp3"を読み込んで再生してみましょう
* サンプリングレートを8000Hzも読み込んで再生してみましょう
* サンプリングレートを変更したら，配列の形状を確認してみよう
* getsamplerate()で元のファイルのサンプリングレートを表示してみよう

### Q.4 音データを可視化してみよう（ノック75）
* librosa.display.waveshowを使って，"音声.mp3"の音声波形を可視化してみましょう（44100/22050/8000Hzそれぞれ可視化してください）
* "携帯電話着信音.mp3"の音源波形を確認してみましょう（44100HzのみでOK）

### Q.5 音データの大きさを取得してみよう（ノック76）
* RMSとは何かを説明してください
* numpyを使って，"音声.mp3"のデータ全体に対してのRMSを計算してください
* "音声.mp3"と"携帯電話着信音.mp3"の時間別のRMSを算出してください
* "音声.mp3"，"携帯電話着信音.mp3"のRMSデータ（全体・時間別ともに）を可視化してください

### Q.6 音データを保存しよう（ノック80）
* "音声.mp3"を，soundfileを使って，WAV形式で音データを保存してください
* 出力したWAVファイルを再度ロードし，その音を確認してください
* **ノック80なので注意してください**

### Q.7 WAVデータを変換してみよう
* wavデータを読み込んでください（wavデータは，dataフォルダを参照）
* pydubのAudiosegmentを使って，flacおよびmp3で音源ファイルを保存（出力してください）
* wav/flac/mp3について説明してください
* wav/flac/mp3のデータの圧縮率を確認してください
* 圧縮率に違いがあれば，その理由を考察してください

### Q.8 周波数スペクトルを表示してみよう（ノック77）

### Q.9 スペクトログラムを可視化してみよう（ノック78）

### Q.10 音の高さや長さを変えてみよう（ノック79）

### Q.11 音データを変換してみよう

# Chapter 3: Preprocessing for Machine Learning
