# Prediction-Of-Cyber-Relationship-Addiction
Extreacted features from social media platform and establish the model(SVM) to detect potential Cyber Relationship Addiction.

# why:
<pre>
隨著網路越來普及，然而台灣成癮人口高達 15% ，但許多人卻渾然不知患有如 此症狀，而我們想從個人的社群網站中資料來預測路成癮 。
</pre>
# what:
<pre>
由中研院社會所提供，包含 1469 位使用者的 社群網站資料以及心理測驗量表。我們從中抽取有用的資料，來提高辨別網路成癮精準度。
</pre>
# How:
<pre>
<img src="https://github.com/marsii1017/Prediction-Of-Cyber-Relationship-Addiction/blob/master/how_to_make.PNG" width="500"> 

1.Facebook:由FacebookFacebook API 抓取使用者社群網站的相關資料。

2.MySQL:利用其函式統計與分析使者不同的特徵，並當作下一步驟地輸入資料。

3.SVM(Support Vector Machine):
從現有資料找出最佳分類方法，來預測新的別
並解決我們樣本數少、非線性、且高維的資料。
</pre>
# Result:
<pre>
<img src="https://github.com/marsii1017/Prediction-Of-Cyber-Relationship-Addiction/blob/master/accuracy.PNG"> 

四種資料處理
TF:把有資料所對應到的feature當作1，反之就當0。
Max:用於資料標準化，把原始資料除上該資料最大值。
In:把有資料所對應到的feature當作０，反之就當１。
Num:原始資料。

</pre>
# Conclusion:
<pre>
<img src="https://github.com/marsii1017/Prediction-Of-Cyber-Relationship-Addiction/blob/master/top_5_features.PNG" width="500"> 
</pre>
