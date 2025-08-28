# Emergency-Logistics-Hub-Optimization-through-Multi-Dimensional-Risk-Assessment
Emergency Logistics Hub Optimization through Multi-Dimensional Risk Assessment

## Overview
This repository contains code for optimizing the location and operation of emergency logistics hubs during disasters by integrating and evaluating multiple risk factors.  
By combining hazards such as earthquakes and floods, population vulnerability, and infrastructure conditions into a “multi-dimensional risk assessment,” the aim is to improve the efficiency and fairness of relief supply distribution.

- Example design of a multi-dimensional risk score (integrating hazards, vulnerability, exposure, and infrastructure)
- Experiments on optimization problems combining site selection and demand allocation
- Visualization (display of hub placement and allocation results on a map)
- Directory structure:
  - `theoretical/` : Theoretical models
  - `experimental/` : Experiments with real data

## Theoretical Models
Code is provided in the `theoretical/` folder.  
 - run_once_v3.sh:  
   Script to execute Virtual_Region_Run_v3.py once  
 - run_multi_v3.sh:  
   Script to execute Virtual_Region_MultiRun_v3.py multiple times

## Experiments with Real Data
Code is provided in the `experimental/code/` folder.   

 - 11_OpenData_Mesh_Generation.ipynb:  
   Processes open data (geojson, shapefile, etc.), performs mesh-based calculations, and saves results as CSV files  
 - 12_Data_Processing.ipynb:  
   Further processes data into the required format  
 - 13_Risk_Calculation.ipynb:  
   Calculates risk levels  
 - 14_Coverage_Problem_Calculation.ipynb:  
   Solves coverage problems

### Preparation
Before running, store the open data and other required datasets in the `data/` folder. Calculation results will be stored in the `results/` folder. Currently, only the final result in HTML format is output.  
Note: A Gurobi license is required to solve the coverage problem. Please prepare your own license.

## License
This code is released under the MIT License.


---

## 概要（Japanese）
このリポジトリは、災害時の緊急物流ハブ（拠点）の配置・運用を、複数のリスク要因を統合的に評価することで最適化するためのコードをまとめたものです。  
地震や洪水といったハザード、人口の脆弱性、インフラ状態などを組み合わせた「多次元リスク評価」を用いて、救援物資の供給効率や公平性を向上させることを目的としています。

- 多次元リスクスコアの設計例（ハザード、脆弱性、露出、インフラを統合）
- 候補拠点の選定と需要割当を組み合わせた最適化問題の実験
- 可視化（地図上での拠点配置や割当結果の表示）
- ディレクトリ構成:
  - `theoretical/` : 理論モデル
  - `experimental/` : 実データでの実験


## 理論モデル
`theoretical/`フォルダ内にコードが記載されています。  
 - run_once_v3.sh:  
Virtual_Region_Run_v3.py を 1回だけ実行するスクリプト  
 - run_multi_v3.sh:  
 Virtual_Region_MultiRun_v3.py を複数回呼び出すスクリプト

 ## 実データでの実験
`theoretical/`フォルダ内に `code/`コードが記載されています。   

 - 11_オープンデータメッシュ化および生成.ipynb:  
 オープンデータがgeojsonやshapeファイルなど、複数の形式になっているため、メッシュごとの計算を行いcsvファイルに保存するプログラム
  - 12_データ整形.ipynb:  
  さらに必要なデータに整形するプログラム
  - 13_リスク計算.ipynb:  
  リスク度を計算するプログラム
  - 14_被覆問題計算.ipynb:  
  被覆問題を計算するプログラム

  ### 事前準備
  事前準備として、オープンデータなど使うデータは`data/`フォルダ内に格納して計算を行います。結果は`results/`フォルダに格納されます。現在は、最終結果のhtmlファイルのみ出力しています。
  また被覆問題を計算する際には、Gurobiのライセンスが必要です。各自用意をしてください。

  ## ライセンス
  このコードはMITライセンスです。
