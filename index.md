---
layout: default
---
# Table of Contents
* [Chapter 1-1: Image Processing and Visualization (pp.224-256)](#chapter-1-1-image-processing-and-visualization)
* [Chapter 1-2: Audio Processing and Visualization](#chapter-1-2-audio-processing-and-visualization)
* [Chapter 1-3: Preprocessing for Machine Learning](#chapter-1-3-preprocessing-for-machine-learning)
* [Chapter 2-1: Image Classification with Transfer Learning and Fine Tuning](#chapter-2-1-image-classification-with-transfer-learning-and-fine-tuning)
* [Chapter 2-5: Image Generation with GANs](#chapter-2-5-image-generation-with-gans)
* [Chapter 2-7: NLP with Transformer](#chapter-2-7-nlp-with-transformer)
* [Chapter 2-6: Anomaly Detection with GANs](#chapter-2-6-anomaly-detection-with-gans)

# Textbook 
* Chapter 1: 下山 輝昌、伊藤 淳二、露木 宏志 著「Python実践 データ加工/可視化 100本ノック」(秀和システム)
* Chapter 2: 小川雄太郎 著「つくりながら学ぶ！PyTorchによる発展ディープラーニング」（マイナビ出版）

# GitHub Repo
* [2022-sem1-rinko](https://github.com/kiyalab-tmu/2022-sem1-rinko)

# Chapter 1-1: Image Processing and Visualization 

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

# Chapter 1-2: Audio Processing and Visualization

* HOMEに課題を解くための雛形をアップロードしましたので下記の手順からダウンロードしてください
1. `git pull`
2. `git checkout main`
3. HOMEディレクトリに、新しいChapができてるので，そこから必要なファイルをコピー（自分のマシンのデスクトップなどに一時的にペースト）
4. `git checkout 自分のブランチ名`で自分のブランチにもどる
5. デスクトップなどに一時的にペーストしたファイルを自分の作業ディレクトリにコピペ
* 発表では、**音声がzoomで聞けるように工夫をお願いします**
* ２つの音声を比較するときは，立て続けに再生して比較するようにしましょう（２つの音声再生の間に，話したりしないようにしましょう）

### Q.1 音データを再生してみよう & 音データを読み込んでみよう（ノック71&72）
* librosaを使用して"音声.mp3"と"携帯電話着信音.mp3"を読み込んでください（ノック72）
* "音声.mp3"と"携帯電話着信音.mp3"を再生してください（ノック71）
* vscode+jupyter環境では音源を再生できないので注意してください（ノック71）
* 音源ファイルのサンプリングレートを確認してください（ノック72）
* ロードした音源を配列に格納し、shape/max/minを確認してください（ノック72）

### Q.2 音データの一部を取得してみよう（ノック73）
* "音声.mp3"と"携帯電話着信音.mp3"を1秒間のデータに切り出してください
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
* フーリエ変換とは何かを説明してください
* "音声.mp3"をフーリエ変換してみましょう
* "音声.mp3"の振幅スペクトルを表示してみましょう
* "音声.mp3"（サンプリングレート22050Hz）の振幅スペクトルを表示してみましょう
* "携帯電話着信音.mp3"の振幅スペクトルを表示してみましょう

### Q.9 スペクトログラムを可視化してみよう（ノック78）
* スペクトログラムとは何かを説明してください
* librosaのライブラリを使って，"音声.mp3"のスペクトログラムを表示してみましょう
* librosaのライブラリを使って，"携帯電話着信音.mp3"のスペクトログラムを表示してみましょう

### Q.10 音の高さや長さを変えてみよう（ノック79）
* librosa.effects.pitch_shiftを使い，"音声.mp3"をピッチのステップ数10で，音を変換してみましょう
* librosa.effects.pitch_shiftを使い，"音声.mp3"をピッチのステップ数-5で，音を変換してみましょう
* librosa.effects.time_stretchを使い，"携帯電話着信音.mp3"をレート数0.5で，音を変換してみましょう
* librosa.effects.time_stretchを使い，"携帯電話着信音.mp3"をレート数2で，音を変換してみましょう

### Q.11 音データを変換してみよう
* VTLNとClippingについて説明してください
* "音声.mp3"を読み込みましょう
* "音声.mp3"にVTLNとClippingを施し，音を再生してみましょう
* VTLNとClippingのパラメータを調節して，結果がどう変わるかを確認してみましょう

# Chapter 1-3: Preprocessing for Machine Learning

* HOMEに課題を解くための雛形をアップロードしましたので下記の手順からダウンロードしてください
1. `git pull`
2. `git checkout main`
3. HOMEディレクトリに、新しいChapができてるので，そこから必要なファイルをコピー（自分のマシンのデスクトップなどに一時的にペースト）
4. `git checkout 自分のブランチ名`で自分のブランチにもどる
5. デスクトップなどに一時的にペーストしたファイルを自分の作業ディレクトリにコピペ

* この章では，タイタニック号の生存状況のデータに基づき，海難事故にあった場合どういった顧客が生存できるのかを予測するモデルを作成します．
### Q.1 機械学習で予測したいデータを分けよう（ノック81）
* seabornのload_datasetを使って，'titanic'のデータセットを読み込みましょう
* 'titanic'のデータセットが，どのようなデータセットなのか簡単に説明してください（各カラムの意味など）
* "目的変数"および"説明変数"とは何かをそれぞれ説明してください
* データセットから，目的変数の列のみを取り出してみましょう

### Q.2 TrainデータとTestデータに分割しよう（ノック82）
* "教師あり学習"と"教師なし学習"とは何かをそれぞれ説明してください（具体的なモデル名が分かればそれを例示してください）
* "Trainデータセット"および"Testデータセット"とは何かをそれぞれ説明してください（"Validationデータセット”など，その他の種類があればそれも説明してください）
* sklearnのtrain_test_splitを使って，データセットをTrainデータ(train_ds)とTestデータ(test_ds)に分割してください
* train_test_splitの仕様（引数など）を簡単に説明してください
* 分割したデータを確認してみましょう

### Q.3 データを機械学習に適した形式へ変換しよう（ノック83）
* 今回のタスクにおける"データリーク"の意味を説明してください（ラベル"survived"とカラム"alive"の関係を例に説明するとよいでしょう）
* 予測モデルを作成するにあたって，必要なデータと不要なデータがそれぞれ何かを説明してください
* dropを用いて，不要な変数名をデータセット(train_ds)から削除しましょう
* "One-hotエンコーディング"および"labelエンコーディング"とは何かをそれぞれ説明してください
* Trainデータセット(train_ds)をone-hotエンコーディングしてみましょう
* Trainデータセット(train_ds)をlabelエンコーディングしてみましょう
* Labelエンコーディングの問題点を挙げてください
* train_dsの"pclass"列をone-hotエンコーディングしてみましょう
* train_dsのTrue/Falseを1/0の数値に変換しましょう

### Q.4 欠損値の処理をやってみよう（ノック88）
* 欠損値とは何かを説明してください
* 欠損値はどのように処理されるのか調べてみましょう（テキスト参照）
* データセット(train_ds)に欠損値がないか確認してください（各カラムごとの欠損値の合計数を確認してください）
* train_dsの欠損値を中央値で補完で補完しましょう
* sklearn.imputeのSimpleImputer使用
* 作成したSimpleImputer(age_imputer)をpickleを用いて保存してください

### Q.5 テストデータの前処理をしよう（ノック90）-パート１-
* Testデータセット(test_ds)の内容を確認しましょう
* Testデータセットの不要な項目を削除しましょう（aliveとembark_town）
* Testデータセットをカテゴリカル変数に変換しましょう
* TestデータセットのTrue/Falseを0/1に変換しましょう
* **今週は、merge関数を用いてTrainデータセットとTestデータセットのカラム違いを解消する必要はありません**
* **今週は、スケーリングを行う必要はありません**
* train_dsの欠損値補完に使用したage_imputerをpickleでロードして、test_dsの欠損値を補完してください（図7-40参照）
* 加工したtest_dsを確認してください

### Q.6 追加課題１: scikit-learn を用いて、モデルを訓練しよう -パート１-
* sklearn.svm.SVCを用いて、モデルを訓練してみましょう
* Trainモデルに対する精度を評価しましょう（モデル名.scoreで評価できるはずです）
* Testモデルに対する精度を評価しましょう
* 今週は，SVMの詳しい説明は不要です（2週間後にまとめて勉強します）

### Q.7 外れ値の検出をしよう（ノック84）
* 外れ値とは何かを説明してください
* 箱ひげとIQRについて簡単に説明してください
* 箱ひげとIQRに基づいて，外れ値に該当するデータの件数を調べましょう

### Q.8 データ分布を見てスケーリング手法を考えよう（ノック85）
* スケーリングとは何か説明してください（特に，正規化・標準化・ロバストスケーリングについて説明してください）
* 正規化・標準化・ロバストスケーリングは，それぞれどのような分布に対して有効なのか説明してください
* train_dsの基本統計量を確認してください
* データ分布の違いをヒストグラムで可視化しましょう
* 統計学の検定（カイ二乗検定とシャピロ-ウィルク検定）について説明してください
* カラムageについて，カイ二乗検定/シャピロ-ウィルク検定のp値を算出してください（さらに，一様性と正規性を判定してください）
* カラムsidspについて，カイ二乗検定/シャピロ-ウィルク検定のp値を算出してください（さらに，一様性と正規性を判定してください）
* カラムparchについて，カイ二乗検定/シャピロ-ウィルク検定のp値を算出してください（さらに，一様性と正規性を判定してください）
* カラムfareについて，カイ二乗検定/シャピロ-ウィルク検定のp値を算出してください（さらに，一様性と正規性を判定してください）

### Q.9 分布に従ってスケーリングをやってみよう（ノック86）
* age，sidsp，parch，fareについて，それぞれどのようなスケーリング手法を行うのが良いか，理由とともに検討してください
* 変数間で分布が同じ場合は，どのようなスケーリングを行うのが良いでしょうか；一方で，変数間で分布が異なる場合は，そのようにスケーリングすれば良いでしょうか（テキストp.306 参照）
* age，sidsp，parch，fareをスケーリングしてください（sklearn.preprocessingにスケーラーがあります）
* スケーリングしたデータの分布をヒストグラムで可視化しましょう

### Q.10 スケーラーを保存しよう（ノック87）
* os.makedirsでフォルダを作成し，スケーラーをpickle形式で保存してください
* スライドでの説明は不要です

### Q.11 学習時のサンプル比率を調整しよう（ノック89）
* 学習時のデータのサンプル比率を調整する必要性について説明してください
* アンダーサンプリングとオーバーサンプリングについて説明してください
* 目的変数のデータ件数を調べてみましょう
* train_dsをアンダーサンプリング/オーバーサンプリングしてください

### Q.12 テストデータの前処理をしよう（ノック90）-パート２-
* Q.5 テストデータの前処理をしよう（ノック90）-パート１-をもう一度行いましょう
* 今週は、カラム違いの解消・スケーリングを**行ってください**
* Pandasのmergeを用いて、TrainデータセットとTestデータセットのカラム違いを解消してください（その原因が分かれば簡単に説明してください）
* カラム解消で用いた、Pandasのmergeについて説明してください（特に、how='left'の部分について）

### Q.13 追加課題２: scikit-learn を用いて、モデルを訓練しよう -パート２-
* SVMとは何か説明してください
* sklearn.svm.SVCを用いて、モデルを訓練してみましょう
* Trainモデルに対する精度を評価しましょう（モデル名.scoreで評価できるはずです）
* Testモデルに対する精度を評価しましょう
* **Q.6 追加課題１の精度と比較してみましょう**

<!---
### Q. 追加課題３: scikit-learn を用いて、色々なモデルを使って学習してみよう
* 線形回帰について説明してください（式・式中の何を学習するのか・最小二乗法との関係性など）
* 回帰と分類の関係性について説明してください（似ているところや違いなども説明してください）
* 生成モデルと識別モデルの違いを説明してください
* パーセプトロン，ロジスティック回帰，SVM（サポートベクトルマシン）についてそれぞれ説明してください
* パーセプトロン・ロジスティック回帰のモデルを使って，Trainモデルに対する精度を評価しましょう
* パーセプトロン・ロジスティック回帰のモデルを使って，Testモデルに対する精度を評価しましょう
-->

# Chapter 2-1: Image Classification with Transfer Learning and Fine Tuning
* 特に指定がない限り、教科書で用いているPytorchのサンプルコードを利用してください
* 「簡単に説明してください」は口頭での説明のみでも構いません
* GPUマシンが混雑している場合は、代表者１〜２名のみが学習を行い、その結果をシェアして発表する形式でも構いません（全員がGPUマシンで学習スクリプトを実行&精度を出すことを行わなくても良いです）
* 学習スクリプトを回さない人でも、コーディングとエラーがないかのチェックは行いましょう

## 学習済みのVGGモデルを使用する方法(p.2~)

### Q.1 データセット
* ImageNetデータセットについて説明してください（クラス数・訓練データ数・ベンチマークなど）
* データセットと関連が深い、ミニバッチ学習について説明してください（なぜそれをやるとよいのかなど）

### Q.2 パッケージのimportとPytorchのバージョン確認
* この課題で利用するパッケージをimportしてください
* Pytorchのパージョンを確認してください
* スライドの説明はなしで構いません

### Q.3 VGG-16の学習済みモデルのロード
* 畳み込み層・プーリング層・全結合層について説明してください
* Dropout層について説明してください
* 活性化関数とReLUについて説明してください
* VGG-16を説明してください（発表年・モデル構造・解決しようとした主な問題など）
* featureモジュールとclassifierモジュールについて説明してください
* VGG-16の学習済みモデルをロードしてください

### Q.4 入力画像の前処理
* Pythonの`__call__()`メソッドについて簡単に説明してください
* torchvisionの`transform.resize()`、`transform.CenterCrop()`、`transform.ToTensor()`、`transform.Normalize()`についてそれぞれ説明してください
* `transform.ToTensor()`を実行すると、PIL形式の入力画像のサイズ（Shape）はどのように変化するか説明してください
* 上記のメソッドを使って、画像を224x244にリサイズおよびセンタークロップしてください
* 画像の色情報を平均(0.485, 0.456, 0.406)、標準偏差(0.229, 0.224, 0.225)で標準化してください
* なぜこのリサイズと標準化が必要なのか説明してください（ヒント：学習済みモデル）
* `./data/goldenretriever-3724972_640.jpg`を読み込み、前処理を行う前後の画像を比較しましょう

### Q.5 出力結果からラベルを予測する後処理
Q.5~Q.6を通しで実行した後で下記の質問を考えるほうが良いと思います
* VGG16の出力サイズ（shape）とそのデータ型は何でしょうか（Q.6を実行するとわかります）
* VGG16の出力をNumPy型に変換してください
* np.argmax()について説明してください
* ラベルとnp.argmax()の関係について説明してください
* `./data/imagenet_class_index.json`を読み込んで、その中身について簡単に説明してください
* 読み込んだjsonを使って、VGGの出力からラベルを得るクラスを作成してください

### Q.6 学習済みVGGモデルで手元の画像を分類
* unsqueeze_()メソッドを行う理由を説明してください（ヒント：バッチサイズ）
* `./data/goldenretriever-3724972_640.jpg`を分類してみましょう

## Pytorchによるディープラーニング実装の流れ（p.14~)

### Q.7 Pytorchによるディープラーニング実装の流れ
* 図1.2.1 (p.15) を参考に，Pytorchによるディープラーニング実装の流れを説明してください
* ネットワークモデルの順伝搬関数 (forward)とは何かを説明してください
* 損失関数について説明してください（計算方法・損失関数の種類など）
* 誤差逆伝播と最適化手法について説明してください（2つの関係性・最急降下法の式・最適化手法の種類など;ただしSGD,Adamなどの個々の詳解は不要です）
* ネットワークモデルの重み(パラメータ)の初期値は，どのような値であるか説明してください
* 過学習とは何かを説明してください
* Validationデータセットとは何か説明してください（Testデータセットとの違いなど）
* early stoppingとは何かを説明してください（導入の目的など）

## 転移学習の実装（p.17~)

### Q.8 転移学習
* 転移学習とは何かを説明してください（ファインチューニングについて調べるとより違いが明確になります）
* 今回のタスクについて説明し、転移学習をどのように適用したかを説明してください

### Q.9 実装の初期設定
* 必要なパッケージをimportしてください（説明は不要）
* 乱数のシードとは何か簡単に説明してください（特に、機械学習との関係について）

### Q.10 Datasetを作成
* 前処理クラスImageTransformを作成してください
* データ拡張（オーギュメンターション）とは何かを説明してください
* `RandomResizedCrop`と`RandomHorizontalFlip`について説明してください
* ImageTransformを施した前後の画像を比較してください
* アリとハチの画像へのファイルパスリストを作成し、それらの画像のDatasetを作成してください

### Q.11 DataLoaderを作成
* 作成したDatasetを使ってDataLoaderを作成してみましょう

### Q.12 ネットワークモデルを作成
* 学習済みのVGG-16モデルをロードしましょう
* 転移学習を行うために、VGG-16モデルを加工してください

### Q.13 損失関数を定義
* softmax関数について説明してください
* Cross Entropy誤差関数について説明してください
* Cross Entropy誤差関数を実装しましょう

### Q.14 最適化手法を設定
* Momentum SGDについて説明してください
* 最適化手法を実装しましょう

### Q.15 学習・検証を実施
* 転移学習を用いて学習・検証を実行してください
* LossとAccuracyを確認してください

## Amazon AWSのクラウドGPUマシンを使用する方法（p.32~)
* 今回はスキップします。興味のある方は読んでみてください。

## ファインチューニングの実装（p.47~)

### Q.16 ファインチューニング
* ファインチューニングとは何かを説明してください

### Q.17 DatasetとDataLoaderを作成
* 転移学習のときと同様に、DatasetとDataLoaderを作成してください（説明は不要）

### Q.18 ネットワークモデルを作成
* 学習済みのVGG-16モデルをロードしましょう
* ファインチューニングを行うために、VGG-16モデルを加工してください

### Q.19 損失関数を定義
* 転移学習のときと同様に、Cross Entropy誤差関数を実装しましょう（説明は不要）

### Q.20 最適化手法を設定
今回は、各層ごとに学習率を変更します
* 学習させる層のパラメータ名を指定し、パラメータごとに各リストに格納してください
* 最適化手法を、パラメータごとに設定してください

### Q.21 学習・検証を実施
* PytorchでGPUを用いた演算を有効化する方法を簡単に説明してください
* ファインチューニングを用いて学習・検証を実行してください
* LossとAccuracyを確認してください

### Q.22 学習したネットワークを保存・ロード
* ネットワークモデルの変数`net`からパラメータを取り出し、保存してください
* 保存した重みを、再度ロードしてみましょう（CPU、GPUでそれぞれ計算されたパラメータの違いに注意してください）

# Chapter 2-5: Image Generation with GANs
* 特に指定がない限り、教科書で用いているPytorchのサンプルコードを利用してください
* GPUマシンが混雑している場合は、代表者１〜２名のみが学習を行い、その結果をシェアして発表する形式でも構いません（全員がGPUマシンで学習スクリプトを実行&精度を出すことを行わなくても良いです）
* 学習スクリプトを回さない人でも、コーディングとエラーがないかのチェックは行いましょう
* 下準備として，make_folders_and_data_downloads.ipynbの実行が必要です

## GANによる画像生成のメカニズムとDCGANの実装(p.242~)
### Q. 準備
* MNISTデータセットについて簡単に説明してください
* 画像生成とは，どのようなタスクであるか説明してください（必要であれば，MNISTデータセットを例にして，画像分類との違いを説明しても良いです）
* GAN(オリジナル）・DCGANについて調査し，そのモデルの構造を説明してください（両者の違いなど）

### Q. Generatorのメカニズム
* DCGANのGeneratorについて説明してください（モデルの構造など）
* 転置畳み込み層について説明してください

### Q. Generatorの実装
* Generatorの入力は何ですか
* Batch Normalizationとは何かについて説明してください
* DCGANのGeneratorを実装してみましょう
* 学習なしでGeneratorから画像を生成し，結果を確認しましょう

### Q. Discriminatorのメカニズム
* Discriminatorについて説明してください（モデルの構造，Discriminatorは何を識別するのか，Generatorとの関係性など）

### Q. Discriminatorの実装
* Discriminatorを実装してみましょう
* 活性化関数であるLeakyReLUについて説明してください

## DCGANの損失関数，学習，生成の実装(p.252~)
### Q. GANの損失関数
* DCGANのDiscriminatorの損失関数について説明してください（テキストの式を解説してください）
* Discriminatorの損失関数を実装してみましょう
* DCGANのGeneratorの損失関数について説明してください（テキストの式を解説してください）
* Generatorの損失関数を実装してみましょう
* DCGANでは，なぜLeakyReLUを活性化関数に利用したのでしょうか？
※ 損失関数は，実装のみで，動作確認はエラーとなりできません（この時点では，入力がないためです）

### Q. DataLoaderの作成
* MNISTデータセットのDataLoaderを実装しましょう（説明は簡単で構いません）
* 今回は，MNISTデータセットの一部のみを使ってGANを学習させる点に注意してください

### Q. DCGANの学習
* DCGANのモデルの初期化を行いましょう
* Generator・Discriminatorを学習するための関数 train_modelを実装しましょう
* 200epochでGenerator・Discriminatorの学習を行ってください
* 生成画像と訓練データを並べて表示し，その結果を確認しましょう
* GANの欠点について調べましょう（２つ以上）
* モード崩壊とは何か説明してください
※ x.float()でデータの型を修正する必要があるかもしれません

## Self-Attention GANの概要(p.265~)
### Q. 従来のGANの問題点
* 転置畳み込み層を用いたGANが持つ問題点について説明してください

### Q. Self-Attentionの導入
* Self-Attentionを導入することで解決される問題は何ですか
* Self-Attentionを式と図を用いて説明してください（特徴マップは1次元化されていることに注意）
* 以上の説明を踏まえて，Self-Attentionの概念を今一度まとめてみましょう→p.269（転置畳み込み層にどのような制約がつくのか，大きなカーネルサイズを用いることがなぜ難しいのかなど）

### Q. 1x1 Convolutions (pointwise convolution)
* pointwise convolutionの処理について説明してください
* Self-Attentionを使用する前段階でpointwise convolutionを用いる２つのメリットを説明してください

### Q. Spectral Normalization
* Spectral Normalizationの概要について説明してください（Batch Normalizationとの違いなど）
* リプシッツ連続性について説明してください

## Self-Attention GANの学習，生成の実装(p.274~)
### Q. Self-Attentionモジュールの実装
* Self-Attentionモジュールを実装してみましょう

### Q. 生成器Generatorrの実装
* Generatorを実装してみましょう
* Spectral Normalization・Self-Attentionに重点をおいて説明してください

### Q. 識別器Discriminatorの実装
* Discriminatorを実装してみましょう
* 上記と同じく，Spectral Normalization・Self-Attentionに重点をおいて説明してください

### Q. DataLoaderの作成
* DataLoaderを実装しましょう（前回と同じなので説明は不要）

### Q. ネットワークの初期化と学習の実施
* Self-Attention GANの損失関数（hinge version of the adversarial loss）について説明してください
* Generator・Discriminatorを学習するための関数 train_modelを実装しましょう
* Self-Attention GANのモデルの初期化を行いましょう（説明は不要）
* 300epochでGenerator・Discriminatorの学習を行ってください
* 生成画像と訓練データを並べて表示し，その結果を確認しましょう
* 生成画像とAttention mapを並べて表示し，その結果を確認しましょう
※ x.float()でデータの型を修正する必要があるかもしれません

# Chapter 2-7: NLP with Transformer
* 特に指定がない限り、教科書で用いているPytorchのサンプルコードを利用してください
* GPUマシンが混雑している場合は、代表者１〜２名のみが学習を行い、その結果をシェアして発表する形式でも構いません（全員がGPUマシンで学習スクリプトを実行&精度を出すことを行わなくても良いです）
* 学習スクリプトを回さない人でも、コーディングとエラーがないかのチェックは行いましょう
* 下準備として，make_folders_and_data_downloads.ipynbの実行が必要です
* **CUDA10.2非対応のGPU(RTX3090など)では，バージョンの問題で学習スクリプトが動作しない可能性があります**（学習以外であれば動作します）
* 必要ライブラリ| janome, torchtext, gensim, spacy（おそらく不要ですがエラーが出ればインストールをお願いします）
* バージョン依存の制約がかなり強いため，自分が試して動いたライブラリのバージョンを列挙します（記載以外でライブラリ〇〇のバージョンを知りたいなどの希望があれば連絡ください）
```
gensim                  4.2.0
Janome                  0.4.2
numpy                   1.19.4
six                     1.15.0
spacy                   3.4.1
spacy-legacy            3.0.10
spacy-loggers           1.0.3
torch                   1.8.2
torchfile               0.1.0
torchtext               0.11.2
zipp                    3.4.0
```
<!---
## 形態素解析の実装(p.328~)
### Q. 機械学習における自然言語処理の流れ
* 自然言語処理（NLP）におけるコーパスとは何か説明してください
* 機械学習における自然言語処理の流れを説明してください（クリーニング，正規化，形態素解析，見出し語化，ストップワード除去，単語の数値化）

### Q. Janomeによる単語分割
* 単語分割のライブラリであるJanomeについて説明いてください
* Janomeを使って単語分割を実装してください（入出力結果や関数群の簡単な説明を含む）

### Q. MeCabによる単語分割
* 単語分割のライブラリであるMeCabおよびNEologdについて説明いてください
* ~~MeCabを使って単語分割を実装してください~~（環境構築が面倒なので今回はスキップします）

## torchtextを用いたDataset、DataLoaderの実装(p.335~)
### Q. 前処理と単語分割の関数を実装
* 使用するデータについて簡単に説明してください
* Janomeを使って、単語分割を行う関数を定義しましょう（説明は不要）
* 前処理を行う関数を定義しましょう（前処理の内容を説明してください）
* 前処理→単語分割を行う関数 tokenizer_with_processing を実装し、結果を確認してください

### Q. 文章データの読み込み
* torchtextのバージョン違いによるエラーを回避する場合は、`import torchtext.legacy as torchtext`で`torchtext`をインポートしてください
* `torchtext.data.Field`の引数や処理の役割を簡単に説明してください
* ./data/text_train.tsvを読み込んで、テキストとラベルからなるデータセット`train_ds`を作成しましょう

### Q. 単語の数値化
* NLPにおける単語の数値化について説明してください（なぜ数値化が必要なのかなど）
* ID0の`<unk>`について説明してください
* ID1の`<pad>`について説明してください
* ボキャブラリーを作成し、結果を確認してください

### Q. DataLoaderの作成
* DataLoaderを作成しましょう（バッチ化してください）
* 出力結果を確認しましょう（ボキャブラリーと照らし合わせて、単語が正しくid化されていることを確認しましょう）

## 単語のベクトル表現の仕組み(p.343~)
### Q. word2vecでの単語ベクトル表現方法
* 単語のid化が抱える２つの問題を説明してください
* 単語のベクトル表現について説明してください（ベクトル化によって，２つの問題はどのように解消されますか）
* CBOWとSkip-gramについて違いなどをそれぞれ説明してください
* Skip-gramの方が，CBOWよりも優れている理由を説明してください

### Q. fastTextでの単語ベクトル表現方法
* fastTextについて簡単に説明してください
* サブワードについて説明してください（英語・日本語での違いなど）

## word2vec，faseTextで日本語学習済みモデルを使用する方法(p.352~)
### Q. word2vecの日本語学習済みモデルを使用する実装
* 注意 | 教科書はMeCabですが，使用しません
* Janomeを使ったtokenizer_with_preprocessing（p335~の7.2）からコピーしてきましょう（説明は不要）
* DataLoaderを以下の通り定義しましょう（説明は不要）
```
import torchtext.legacy as torchtext

# tsvやcsvデータを読み込んだときに、読み込んだ内容に対して行う処理を定義します
# 文章とラベルの両方に用意します

max_length = 25
TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing,
                            use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length)
LABEL = torchtext.data.Field(sequential=False, use_vocab=False)


# フォルダ「data」から各tsvファイルを読み込みます
train_ds, val_ds, test_ds = torchtext.data.TabularDataset.splits(
    path='./data/', train='text_train.tsv',
    validation='text_val.tsv', test='text_test.tsv', format='tsv',
    fields=[('Text', TEXT), ('Label', LABEL)])
```
* word2vecを実装し，簡単にコードの説明をしてください
* 1単語あたりの次元数と，単語数を調べてください
* ベクトル化したバージョンのボキャブラリーを作成し，結果を表示してみましょう
* ボキャブラリー単語の順番を確認してください（**この確認は重要です**）
* `姫 - 女性 + 男性`のベクトルを計算し，`王子`とのベクトルが近くなることを確認しましょう
* 注意 | **Janomeの使用により，単語のインデックスが教科書のものとずれていますので，インデックスを確認して修正してください**
* `王子`以外にも，`女王`，`王`，`機械学習`のベクトルと比較しましょう

### Q. fastTextの日本語学習済みモデルを使用する実装
* fastTextを実装し，簡単にコードの説明をしてください（以下，word2vecと同様です）
* 1単語あたりの次元数と，単語数を調べてください
* ベクトル化したバージョンのボキャブラリーを作成し，結果を表示してみましょう
* ボキャブラリー単語の順番を確認してください（**この確認は重要です**）
* `姫 - 女性 + 男性`のベクトルを計算し，`王子`とのベクトルが近くなることを確認しましょう
* 注意 | **Janomeの使用により，単語のインデックスが教科書のものとずれていますので，インデックスを確認して修正してください**
* `王子`以外にも，`女王`，`王`，`機械学習`のベクトルと比較しましょう

## IMDbのDataLoaderを実装(p.359~)
### Q. IMDbデータセットをtsv形式に変換
* IMDbデータセットについて簡単に説明してください
* csv，tsvについて簡単に説明してください
* IMDbデータセットを，tsv形式に変換するプログラムを実装してください（処理の内容なども説明してください）

### Q. 前処理と単語分割の関数を定義
* 前処理と単語分割の関数を実装してください
* どのような文字整形を行なったかなど説明してください

### Q. Dataset・ボキャブラリー・DataLoaderの作成
* torchtextのバージョン違いによるエラーを回避する場合は、`import torchtext.legacy as torchtext`で`torchtext`をインポートしてください
* torchtext.data.Fieldの使用を簡単に説明してください
* torchtext.data.Fieldを用いて，tsvを読み込んでTEXTとLABELに施す処理を定義してください
* torchtext.data.TabularDataset.splitsを使って，Datasetを作成しましょう
* Datasetの数が25000であることを確認しましょう
* Datasetを訓練・検証用のデータセットに分割しましょう（それぞれ20000件と5000件）
* ボキャブラリーを作成しましょう
* 1単語あたりの次元数と，単語数を調べてください
* ボキャブラリーの内容を確認しましょう
* DataLoaderを作成し，作成されたDataLoaderからの出力を確認しましょう
* DataLoaderからの出力がベクトル表現でない理由を説明してください

## Transformerの実装(p.367~)
### Q. NLPとTransformerの関係
* テキスト分類など，NLPにはどのようなタスクがあるか調べましょう
* 今回は，どのNLPタスクに挑戦しますか？
* 画像データと言語データの違いについて説明してください
* 再帰的ネットワーク（RNN，LSTM）について説明し，その問題点もあげてください
* 言語データへCNNを使用する方法について説明し，その問題点もあげてください

### Q. Transformerのネットワーク構造と実装
* 今回使用するTransformerの全体像を説明してください（入出力なども含めて）
* Embedderモジュールについて説明してください(pp.369~371)
* Embedderモジュールを実装してみましょう(pp.371~372)
* PositionalEncoderモジュールについて説明してください(pp.373~374)
* Transformerモジュールについて，maskも含めて説明してください(pp.375~378)
* ClassificationHeadモジュールについて簡単に説明してください(pp.379)
* 各モジュールを組み合わせて，Transformer（全体）を実装してください

## Transformerの学習・推論，判定根拠の可視化を実装(p.382~)
### Q. DataLoaderとTransformerモデルの用意
* DataLoaderとTransformerを実装してください（過去に使用したコードを使い回せます）

### Q. 損失関数と最適化手法
* 損失関数と最適化手法を定義（実装）してください

### Q. 訓練と検証の関数の実装と実行
* 訓練および検証を行う関数を実装しましょう
* 実行結果（損失や精度など）を表示しましょう

### Q. テストデータでの推論と判定根拠の可視化
* 訓練したTransformerを用いて，テストデータでの正解率を確認しましょう

### Q. Attentionの可視化で判定根拠を探る
* 説明可能な人工知能（XAI）について簡単に説明してください
* Attentionの可視化の関数を実装してください（説明は不要）
* indexを制御して，テストデータミニバッチ中のあるデータに対するAttentionの結果を可視化しましょう
* 正解できたテストデータ，正解できなかったテストデータをそれぞれ可視化してみましょう
* 可視化した２つのテストデータについて簡単に分析してみましょう
--->
<!--- ViTについてやる？ --->

# Chapter 2-6: Anomaly Detection with GANs
* 特に指定がない限り、教科書で用いているPytorchのサンプルコードを利用してください
* GPUマシンが混雑している場合は、代表者１〜２名のみが学習を行い、その結果をシェアして発表する形式でも構いません（全員がGPUマシンで学習スクリプトを実行&精度を出すことを行わなくても良いです）
* 学習スクリプトを回さない人でも、コーディングとエラーがないかのチェックは行いましょう
* 下準備として，make_folders_and_data_downloads.ipynbの実行が必要です

<!---
## GANによる異常画像検知のメカニズム(p.290~)
### Q. 準備
* 異常画像検出とは何かと，その重要性について説明してください

### Q. AnoGANの概要
* AnoGANの処理の流れを説明してください
* AnoGANの特徴をまとめましょう（どのようなGANがAnoGANとして利用できるのか，学習済みの識別器Dだけで異常検出を行うことは可能か，など）

## AnoGANの実装と異常検知の実施(p.294~)
### Q. DCGANの学習
* DCGANを使用して，AnoGANの実装してください
* AnoGAN実装にあたって，DCGANの変更した点があれば，それを説明してください

### Q. AnoGANの生成乱数zの求め方
* テスト画像に最もよく似た生成画像を生み出すノイズzの求め方について説明してください
* 生成乱数zを求めるプログラムを実装しましょう

### Q. AnoGANの損失関数
* AnoGANの損失関数の概要を説明してください
* Residual lossとdiscrimination lossについて，それぞれ説明してください
* AnoGANの損失関数を実装してください

### Q. AnoGANの学習の実装と異常検知の実施
* 今回の実験設定（どのような画像を異常画像とするのか，など）について説明してください
* 実験設定を踏まえて，テスト用のDataLoaderを実装してください（説明は簡単で構いません）
* テストデータを表示するプログラムを実装しましょう
* テスト画像に最もよく似たzを生成しましょう（上の設問で作成したプログラムを使用）
* 生成したノイズzを使って，Generatorから画像を生成し，lossと生成画像を表示させましょう
* AnoGANの問題点を調査しましょう

## Efficient GANの概要(p.303~)
### Q. Efficient GAN
* Efficient GANのテクニックの概要について説明してください

### Q. エンコーダEを作る方法，後から作る作戦がよくない理由
* 生成器Gを学習させた後に，その逆変換を学習するモデルEを構築することが難しい理由を説明してください
* 必要であれば，VAEについても調査し，簡単に説明してください

### Q. エンコーダEをGANと同時に作る方法
* BiGAN(Bidirectional GAN)について，Efficient GANとの関係性も含めて説明してください
* 識別器Dの損失関数について式を含めて説明してください
* 生成器Gの損失関数について式を含めて説明してください
* エンコーダEの損失関数について式を含めて説明してください

## Efficient GANの実装と異常検知の実施(p.311~)
### Q. GeneratorとDiscriminatorの実装
* Efficient GANの生成器Gを実装し，動作を確認しましょう
* Efficient GANの識別器Dを実装し，動作を確認しましょう
* これまでに実装してきたGANと大きく異なる点があれば，それを説明してください

### Q. Encoderの実装
* エンコーダEを実装し，動作を確認しましょう

### Q. DataLoaderの作成
* DataLoaderを実装しましょう（6.2節とほぼ同じなので説明は不要）

### Q. Efficient GANの学習
* 生成器G・識別器D・エンコーダEを学習するための関数 train_modelを実装しましょう
* 学習率の設定値とその値を選んだ理由について説明してください
* 生成器Gから画像を生成し，教師データと生成データを表示し，比較しましょう

### Q. Efficient GANによる異常検知
* Efficient GANによる異常検知を行うために，異常度を計算する関数(Anomaly_score)を改変してください
* テスト画像とEfficient GANで再構成した画像を比較しましょう
* 異常度を計算し，比較しましょう
* AnoGANと比較して，生成ノイズzの計算時間はどのように変わりましたか
※ x.float()でデータの型を修正する必要があるかもしれません
--->
