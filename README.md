# Fake News and Misinformation Detection

Projekat iz Mašinskog učenja, Matematički fakultet, 2025/26.
Bogdan Tomić 1040/2025 i David Aksović 1048/2025.

## O čemu se radi

Za dati tekst novinskog članka treba odrediti da li je lažan. Uporedili smo
trinaest modela, od najprostije bazne linije do dotreniranog DistilBERT-a.

Skup je [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
sa Kaggle-a, poznat i kao ISOT. Prave vesti su preuzete sa Reuters servisa, a
lažne sa političkih blogova.

Ta podela izvora je ispala važnija nego što smo mislili na početku, pa dobar deo
rada ide na to da izmerimo koliko rezultat zavisi od modela, a koliko od same
građe skupa.

## Kako se pokreće

```
1. data_prep.ipynb            preuzimanje, ciscenje, snimanje
2. program.ipynb              klasicni modeli i neuronske mreze
3. colab_transformeri.ipynb   DistilBERT i embedinzi, treba GPU
```

Redosled je bitan jer svaki notebook koristi ono što je prethodni napravio.
Zavisnosti se instaliraju sa `pip install -r requirements.txt`, a za preuzimanje
skupa treba `kaggle.json` u `~/.kaggle/`.

Treći notebook radi na Google Colab-u ili bilo kojoj mašini sa GPU-om. Ako
`data/news_colab.parquet` postoji, čita ga direktno; ako ne, ponudi upload.

Na procesoru sa četiri jezgra ceo `program.ipynb` traje oko 75 minuta, najviše
zbog BiLSTM-a, plus još oko 15 minuta za pretragu hiperparametara u sekciji 14.
Ta sekcija zavisi samo od učitanog `df`, pa može da se pokrene i sama. Dotreniranje DistilBERT-a na T4 kartici traje 6 minuta, a na
procesoru bi trajalo 47 sati, pa smo ga zato izdvojili u zaseban notebook.

## Priprema podataka

Prvo smo trenirali na sirovom tekstu i dobili gotovo savršene rezultate, pa smo
krenuli da tražimo razlog. Ispostavilo se da ih ima nekoliko.

Skoro svaki članak iz `True.csv` počinje sa `WASHINGTON (Reuters) -`, a nijedan
iz `Fake.csv` nema ništa slično. Isto važi za dane u nedelji, koje Reuters
navodi u svakom izveštaju, i za šablon blog platforme tipa `featured image via`.
Model to nauči umesto jezika, pa smo ih uklonili. Reči stila kao `you`, `just` i
`like` smo zadržali, jer one jesu odlika senzacionalističkog pisanja.

Zatim smo videli da 820 članaka nema tekst, samo naslov i sliku, i da je od njih
819 lažno. I to je prečica, pa su izbačeni.

Najveći problem su bili duplikati. Pre nego što smo ih uklonili, **18.9 % test
članaka imalo je identičnu kopiju u trening skupu**, što je podizalo tačnost za
oko pola procentnog poena.

```
44 898  polazni skup
   820  clanci bez teksta
 5 564  duplikati
38 514  konacno, 21 189 pravih i 17 325 laznih
```

Bitan detalj je redosled: čišćenje ide **pre** stemminga. Obrnuto ne radi, jer
Porter stemmer pretvori `reuters` u `reuter` i `january` u `januari`, pa ih
obrasci više ne pogađaju.

Skup se zatim deli 75/25 sa `random_state=42`. **Svi modeli koriste isti test
skup od 9 629 članaka**, pa su brojevi direktno uporedivi. Semena su fiksirana
za `random`, `numpy` i TensorFlow, tako da se rezultati ponavljaju.

## Rezultati

| Model | Tačnost | Preciznost | Odziv | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| DistilBERT (dotreniran) | 0.9973 | 0.9981 | 0.9958 | 0.9970 | 0.9998 |
| CNN + LSTM (hibrid) | 0.9910 | 0.9885 | 0.9915 | 0.9900 | 0.9985 |
| BiLSTM | 0.9891 | 0.9873 | 0.9885 | 0.9879 | 0.9979 |
| CNN | 0.9888 | 0.9909 | 0.9841 | 0.9875 | 0.9990 |
| TF-IDF + LogReg (bigrami) | 0.9830 | 0.9867 | 0.9753 | 0.9810 | 0.9978 |
| TF-IDF + LogReg | 0.9805 | 0.9839 | 0.9725 | 0.9782 | 0.9974 |
| Linearni SVM | 0.9790 | 0.9802 | 0.9730 | 0.9766 | |
| Logistička regresija | 0.9756 | 0.9796 | 0.9658 | 0.9727 | |
| Kernelizovani SVM (RBF) | 0.9725 | 0.9761 | 0.9624 | 0.9692 | |
| Embedinzi + LinearSVC | 0.9464 | 0.9454 | 0.9349 | 0.9401 | |
| Embedinzi + LogReg | 0.9400 | 0.9420 | 0.9233 | 0.9326 | 0.9839 |
| KNN (k = 5) | 0.8547 | 0.9146 | 0.7467 | 0.8222 | |
| Dummy, većinska klasa | 0.5502 | | | | 0.5000 |

Preciznost i odziv se odnose na klasu lažna vest. Kernelizovani SVM je zbog
kvadratne složenosti treniran na 8 000 primera, pa ga treba čitati uz tu ogradu.

DistilBERT je pogrešio na 26 od 9 629 članaka. Prve tri mreže se razlikuju za
manje od dve desetine poena, što je razlika od dvadesetak članaka, pa ih treba
smatrati izjednačenim.

### Hiperparametri

Probali smo `GridSearchCV` sa trostrukom unakrsnom validacijom.

```
KNN                    k = 21   cv 0.8570   test 0.8526
Linearni SVM           C = 1    cv 0.9764   test 0.9790
Logisticka regresija   C = 10   cv 0.9758   test 0.9801
```

Nije mnogo promenilo. Za SVM je podrazumevano `C = 1` već bilo najbolje, KNN sa
`k = 21` daje isto kao sa `k = 5`, a jedino se logistička regresija popravila,
sa 0.9756 na 0.9801.

### Šira pretraga hiperparametara

Posle predaje smo, na predlog asistenta, dodali širu pretragu na jednom
jednostavnom modelu, logističkoj regresiji nad TF-IDF vektorima (sekcija 14 u
`program.ipynb`). Ovde nema unakrsne validacije, skup je odmah na početku
podeljen na trening (60 %, 23 108 članaka), validacioni (20 %, 7 703) i test
(20 %, 7 703), stratifikovano. Sve kombinacije se biraju po validacionom skupu,
a test se koristi jednom, na kraju.

Pretraga ide u dva koraka. Prvo se traže parametri vektorizatora sa
podrazumevanom logističkom regresijom, 32 kombinacije:

```
max_features   1000, 5000, 20000, 50000
ngram_range    (1, 1), (1, 2)
min_df         1, 5
sublinear_tf   False, True
```

Zatim se na najboljem vektorizatoru traže parametri klasifikatora, 28
kombinacija:

```
C             0.001, 0.01, 0.1, 1, 10, 100, 1000
penalty       l1, l2
class_weight  None, balanced
```

Najbolje po validaciji:

```
TF-IDF   max_features 5000, ngram (1, 2), min_df 1, sublinear_tf True   validacija 0.9805
LogReg   C = 10, l2, class_weight balanced                              validacija 0.9856

trening    0.9979
validacija 0.9856
test       0.9860

podrazumevani model (max_features 5000, C = 1) na istom testu   0.9740
```

Šta se vidi. Bigrami i `sublinear_tf` pomažu najviše od parametara
vektorizatora, oko pola poena svaki. Veličina rečnika iznad 5 000 ne pomaže,
sa 50 000 obeležja validacija je čak malo niža, a trening viša. `min_df` ne
menja ništa jer `max_features` ionako zadrži samo česte reči. Kod
klasifikatora najviše znači `C`: sa 0.001 model ne nauči ništa (0.55, kao
Dummy), do 10 raste, a od 100 naviše je tačnost na treningu 1.0000 dok
validacija polako pada, što je čist primer preprilagođavanja. `l2` je stalno
malo bolji od `l1`, a `class_weight` menja tek treću decimalu.

Ukupno je pretraga donela 1.2 poena u odnosu na podrazumevani model, sa 0.9740
na 0.9860, i validacija se poklapa sa testom do pola desetinke, pa izbor po
validaciji nije ulepšao rezultat. Grafici sa krivama po `C` i po veličini
rečnika su u notebook-u.

### Preprilagođavanje

```
CNN         najbolja epoha 5 od 7    train 0.0103   val 0.0264
BiLSTM      najbolja epoha 7 od 9    train 0.0198   val 0.0463
CNN + LSTM  najbolja epoha 2 od 4    train 0.0436   val 0.0403
```

Sve tri mreže krenu da se preprilagođavaju posle najbolje epohe, ali ih rano
zaustavljanje preseče i vrati težine. Validaciona i test tačnost se poklapaju do
0.0035, pa u konačnim modelima preprilagođavanja nema. Hibrid je zanimljiv jer
je pobedio sa svega dve epohe, i kod njega je validacioni gubitak čak niži od
trening.

### Analiza grešaka

```
pogresnih: 202 od 9629
  lazne  117
  prave   85

duzina:  pogresni 2542 znaka,  tacni 2388
kategorije:  politicsNews 68, politics 46, News 27, left-news 25
```

Greške su blago pomerene ka lažnim vestima i ka dužim člancima. Najviše ih je u
političkim kategorijama, gde su stilovi dva izvora najsličniji.

## Šta smo zaključili

**Visok rezultat govori o lakoći zadatka, ne o kvalitetu modela.** Pravilo „ako
tekst ne sadrži reč Reuters, proglasi ga lažnim" daje **0.9933** bez ikakvog
treniranja. Ono doduše čita sirov tekst dok modeli rade nad očišćenim, ali baš
zato pokazuje koliko je signala bilo u samom potpisu agencije.

**Uklanjanje tragova izvora košta malo.** Isti model i ista podela, menja se samo
ulazna kolona:

```
                  LogReg   LinearSVC
originalni tekst  0.9873    0.9952
ocisceni tekst    0.9830    0.9916
```

**Embedinzi su izgubili od TF-IDF**, 0.9400 naspram 0.9830, što nas je
iznenadilo. Razlog je dvostruk. Model `all-MiniLM-L6-v2` prima najviše 256
tokena, a naši članci imaju prosečno 514, pa ne vidi ni pola teksta. I treniran
je da prepozna da li su dva teksta o istoj temi, a lažna i prava vest o istoj
osobi jesu o istoj temi. To je ovde pogrešna vrsta sličnosti.

**Model ne prenosi znanje između kategorija.** Kad treniramo na `politicsNews` i
`News`, a testiramo na `worldnews` i `politics`, linearni SVM pada sa 0.9790 na
**0.9268**, a logistička regresija sa 0.9756 na 0.9247.

## Kako ovo stoji prema sličnim radovima

Skup je često korišćen, pa postoji dosta radova sa istim modelima. Uobičajen
pristup je da se nabroji nekoliko klasifikatora, istrenira i prikaže tabela
tačnosti, koja skoro uvek završi oko 0.99.

Uklanjanje dateline je poznata praksa. U literaturi se navodi da 100 % pravih
vesti počinje oblikom `grad (Reuters) -` i da to mora da se očisti jer bi
„neistinito podiglo tačnost svakog modela". Ono što se ređe radi je da se izmeri
koliko je taj artefakt vredeo, pa smo to uradili.

Poznato je i da je skup zasićen: modeli koji ovde pređu 99 % osetno padaju na
težim skupovima kao što su LIAR ili FakeNewsNet. Nismo imali drugi skup za
proveru, pa smo testirali generalizaciju između kategorija unutar istog skupa,
što daje sličan zaključak u manjem obimu.

U odnosu na tipičan projekat sa ovim skupom, ovde dodatno postoje bazna linija
od jedne reči, deduplikacija sa izmerenim efektom, analiza grešaka i poređenje
zamrznutih embedinga sa TF-IDF.

## Ograničenja

Skup ima samo dva izvora. Model zato uči da razlikuje dva načina pisanja, a ne
istinu od laži, i ne znamo kako bi radio na vesti sa trećeg portala. Prava
provera bi tražila prikupljanje novih članaka, što nismo stigli.

Isti test skup je korišćen za trinaest modela, pa je izbor najboljeg po njemu
blago optimistički.

Kernelizovani SVM je treniran na manjem uzorku, pa nije potpuno uporediv sa
ostalima.

## Sadržaj

```
data_prep.ipynb             priprema podataka
program.ipynb               klasicni modeli, mreze, evaluacija, pretraga hiperparametara
colab_transformeri.ipynb    DistilBERT i embedinzi
rezultati_colab.json        rezultati sa GPU masine
requirements.txt            zavisnosti
```

Folder `data/` nije u repozitorijumu, pravi ga `data_prep.ipynb`.
