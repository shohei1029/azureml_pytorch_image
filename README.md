# Azure Machine Learningを使用したPyTorchモデルの分散深層学習

本コンテンツは、ローカルの Visual Studio Code (VSCode) 上から Azure Machine Learning を使用して PyTorch モデルの分散深層学習を行うハンズオンコンテンツです。

![](./img/environment_image.png)

<br></br>

![](./img/AML_flow_image.png)

## 対象者イメージ
現在Pythonを使った機械学習・データ分析の経験があり、クラウド上での機械学習やAzure Machine Learningの利用に興味がある方。クラウド上での分散深層学習を体験したい方。

## 前提条件
本ハンズオンコンテンツでは下記環境を前提としています。
- [Anaconda](https://www.anaconda.com/products/individual)  
    Python本体に加え、科学計算やデータ分析に使えるライブラリ群、仮想環境作成機能が提供されているパッケージ
- [Visual Studio Code (VSCode)](https://azure.microsoft.com/ja-jp/products/visual-studio-code/)  
    様々なOSで動作する、機能性と拡張性に優れたオープンソースのプログラミングエディタ
- [VSCode Python 拡張機能](https://marketplace.visualstudio.com/items?itemName=ms-python.python)  
    VSCodeでPythonのコード補完、デバッグ、コード整形、テスト等々を可能にする拡張機能
- [VSCode Jupyter 拡張機能](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)  
    VSCodeでJupyter notebookをサポートする拡張機能。(Python以外の言語でも利用可能)
- [Azure Machine Learning ワークスペース作成](https://docs.microsoft.com/ja-jp/azure/machine-learning/how-to-manage-workspace?tabs=azure-portal)
- GPUインスタンスのクォーターが十分存在すること
    - 手順
        - [リージョンごとにクォータの引き上げを要求する](https://docs.microsoft.com/ja-jp/azure/azure-portal/supportability/regional-quota-requests#request-a-quota-increase-by-region-from-help--support)
    - 申請内容
        - クォータの種類：Machine Learning サービス
        - 場所：(Azure MLワークスペースと同一リージョン)
        - VMシリーズ：NC Series (又はNCSv3 Series等)
        - 新しい vCPU の制限：最低12以上


## 実行手順
本リポジトリを`git clone`するか、ZIPファイルとしてダウンロードしてご利用ください。

### 環境準備
`./setup.ipnb`を実行します。

### Azure MLでの学習とデプロイ (メインコンテンツ)
`./examples/distributed-pytorch-with-distributeddataparallel.ipynb`を実行します。

## 参考情報
- [VSCode の Azure ML 拡張機能チュートリアル](https://docs.microsoft.com/ja-jp/azure/machine-learning/tutorial-setup-vscode-extension)
- [Azure Machine Learning を使用して PyTorch モデルを大規模にトレーニングする](https://docs.microsoft.com/ja-jp/azure/machine-learning/how-to-train-pytorch#distributeddataparallel)
- Horovod を使用した分散深層学習を行うサンプルノードブック
[Distributed PyTorch with Horovod](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/pytorch/distributed-pytorch-with-horovod/distributed-pytorch-with-horovod.ipynb)
    ※本コンテンツでは分散学習を行うためにPyTorch の DistributedDataParallel 機能を使用しています。
- [Azure Machine Learningのサンプルノートブック集 (英語)](https://github.com/Azure/MachineLearningNotebooks)
    PyTorch以外のライブラリを使用した場合を含め、様々なシナリオについてのサンプルノートブックがまとめられています。
- [Machine Learng Practices and Tips](https://azure.github.io/machine-learning-best-practices/#/)

### 関連ノートブック
- モデル学習部分の
[Distributed PyTorch with DistributedDataParallel](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/pytorch/distributed-pytorch-with-distributeddataparallel/distributed-pytorch-with-distributeddataparallel.ipynb)
- モデルデプロイ部分
[Train, hyperparameter tune, and deploy with PyTorch](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/pytorch/train-hyperparameter-tune-deploy-with-pytorch/train-hyperparameter-tune-deploy-with-pytorch.ipynb)
