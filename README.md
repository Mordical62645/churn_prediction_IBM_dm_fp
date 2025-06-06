# churn_prediction_IBM_dm_fp

Final project for Advanced Data Mining.

## Objective
* Predict customers likely to stop using telecom services to target retention efforts.

## Data
* Customer usage, billing, complaint records, demographic info.

## Procedure Used
* Data Balancing (SMOTE for imbalanced data)
* Classification Algorithms (Logistic Regression, XGBoost)
* Feature Importance Analysis

## Outcome
* Proactive retention campaign to reduce churn rate.

## Setup

### For Google Colab:
Run the first cell to download the `IBM.csv` dataset
``` bash
     !gdown --fuzzy "https://drive.google.com/file/d/1rYFumAaLcacQb59IYC-g_8douKWOIkTi/view?usp=sharing"
```

### Local
After cloning the repository, create a virtual environment:
```bash
    py -m venv .venv
```
Activate the virtual environment:
```bash
    .venv/Scripts/activate
```
Install the libraries from `requirements.txt`
```bash 
    pip install -r requirements.txt
```
Some requirements can't be installed all at once. Try manually for example:
```bash
    pip install xgboost 
```

## For Flask x React integration (para sa mga kagroup kong KUMAG)
Refer to `structure.txt` for the project folder structure.

## Dyosa - Skusta Clee Lyrics
Oh, ako'y pinagpala natagpuan ko na

Ang hinahanap kong dalaga

Siya ang dyosa, dyosa, dyosa ng buhay ko

Dyosa, dyosa, dyosa ng buhay ko

Sounds like la-la-la-la

Mukang hindi ko na kailangan ng Maria-Juana

Siya ang dyosa, dyosa, dyosa ng buhay ko

Dyosa, dyosa, dyosa ng buhay ko

Akala ko wala ng kwenta ang buhay ko

Pero sa 'yo ay nagkakulay 'to

Pagdating sa 'yo tanggal dalawang sungay ko

'Yung dating matigas nagiging lantang gulay 'to

Hindi man ako kasing matcho ni Johnny Bravo

Pero itataya ko sa 'yo lahat, pati bato

Oo, hindi ako biniyayaan maging gwapo

Pero, I promise to you, baby, hindi ako ugaling aso

If you ask me

'Di baleng ma-inlove sa 'yo si Skusta Clee

Alam ko kasi 'di ka basta-basta, bhie

Tapos 'yung dating mo napakalakas pa, see

Isa pa, goddamn, you so hot like fire

You so fly, lagpas sky

Gusto kita and I won't deny, oh

'Pag napasa 'kin ka, jusko po, Inay

Oh, ako'y pinagpala natagpuan ko na

Ang hinahanap kong dalaga

Siya ang dyosa, dyosa, dyosa ng buhay ko

Dyosa, dyosa, dyosa ng buhay ko

Sounds like la-la-la-la

Mukang hindi ko na kailangan ng Maria-Juana

Siya ang dyosa, dyosa, dyosa ng buhay ko

Dyosa, dyosa, dyosa ng buhay ko

Nakilala ko na ang babaeng

Tinitingala ko sa tuwing ako ay tambay

Ramdam ko talaga ang mga bagay

Lalo na sa panahon ako'y nakikibagay

Minsan lang tumagay, napadaan pa 

Sabi nila sa 'kin lalo't na kita ko na

Pagkakataon ko na, makausap ko siya

Ako ay na taranta, teka lang, ano, eh

Dami ko naisip, pati panaginip na

Akala ko'y hihinto ngunit isa lang pala

Ang ibig sabihin na kami pala ang magtatagpo

Sana dito mabuo mapatunayan ko sa 'yo

Na ikaw lang ang dyosa

Mahirap hanapin mula sa 'king puso

Nag-wawaangang kaya mga dahon?

'Di ko na kailangan kausapin

Boss d'yan ka lang 'wag ka aalis

Pagdaan mo lagi na, nandito na ko

Miss, 'di kita pababayaan, lagi aalagaan

Bukod tangi kang diwata, na aking mamahalin

Sa palagi kong tingin, ako'y natataranta

Dami mo kasing alam kapag nandiyan ka

Duda na 'ko, 'di na 'ko makanta

Ako sa labas, sa bahay kasi dumaan ka 'di ba?

Tayong dalawa dito sa panaginip

Papunta sa langit at puno ng sigla

Mga bituin, mga labi sana mapasa 'kin, baka umalis pa

'Di ko hahayaan balik-balikan ka

Ng mga titig na nakakatunaw sa ilaw na gamit ko

Sa buong mundo wala nang makakaagaw dahil

Oh, ako'y pinagpala natagpuan ko na

Ang hinahanap kong dalaga

Siya ang dyosa, dyosa, dyosa ng buhay ko

Dyosa, dyosa, dyosa ng buhay ko

Sounds like la-la-la-la

Mukang hindi ko na kailangan ng Maria-Juana

Siya ang dyosa, dyosa, dyosa ng buhay ko

Dyosa, dyosa, dyosa ng buhay ko

Sana'y nakikinig ka, iniibig 

'Wag ka nang mag-alala dahil sa kanila ay ibang-iba ka

'Di na dapat magtaka kung ba't gusto kita

Tinawag kitang dyosa dahil ang iba'y sinasamba ka

Oh, ako'y pinagpala natagpuan ko na

Ang hinahanap kong dalaga,

Siya ang dyosa, dyosa, dyosa ng buhay ko

Dyosa, dyosa, dyosa ng buhay ko

Sounds like la-la-la-la

Mukang hindi ko na kailangan ng Maria-Juana

Siya ang dyosa, dyosa, dyosa ng buhay ko

Dyosa, dyosa, dyosa ng buhay ko

Dyosa, dyosa, dyosa ng buhay ko

Dyosa, dyosa, dyosa ng buhay ko

Dyosa, dyosa, dyosa ng buhay ko

Dyosa, dyosa, dyosa ng buhay ko
