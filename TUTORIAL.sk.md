# Tutoriál · 07 · P2: Odchod zákazníkov (churn), tabuľkové ML od začiatku do konca

🇬🇧 [English version](TUTORIAL.md) · 📄 [README (výsledky)](README.sk.md)

Tento tutoriál ide po workflowe v poradí, v akom vznikal: dáta → EDA → baseliny → pipeline → modely a ladenie → zámerný leakage → prah z nákladov → kalibrácia → vysvetlenia → API. Každý krok hovorí, **čo** sa spravilo, **prečo** a **ako** to funguje. README obsahuje výsledné čísla.

**Cieľ:** nie len „natrénovať model“, ale odpovedať na biznisovú otázku: *ktorí zákazníci majú dostať retenčnú ponuku za 100 €?* A spraviť to bez toho, aby si sám seba oklamal.

**Potrebuješ:** projekt 06 (prečo PR-AUC, nie accuracy ani ROC-AUC). Knižnice: pandas, scikit-learn, LightGBM, Optuna, SHAP, FastAPI. Na Macu LightGBM potrebuje aj `brew install libomp`.

---

## Krok 1 · Načítaj dáta a vyrob pár príznakov ([`data.py`](data.py))

```python
df = pd.read_csv(download())                               # IBM Telco, 7043 zákazníkov
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
y = (df.pop("Churn") == "Yes").astype(int)
X = add_features(df.drop(columns="customerID"))
```

**Prečo každý riadok:**
- `TotalCharges` je uložené ako text a 11 úplne nových zákazníkov tam má `" "` (ešte nedostali faktúru). `errors="coerce"` z toho spraví NaN a neskôr to vyrieši imputer. Potichu doplniť 0 by bolo rozhodnutie urobené bez pozretia.
- **Zahoď `customerID`:** ID nenesie žiadny signál o odchode, len spôsob, ako si model zapamätá jednotlivé riadky.
- `y` je 1 pre odídených. Odišlo 26,5 % zákazníkov: nevyvážené, takže accuracy klame (projekt 06).

**Vyrobené príznaky** (`add_features`): počet služieb, priemerná minulá mesačná platba (`TotalCharges / tenure`) a `charge_change` = dnešná platba mínus tento priemer (nedávne zdraženie). **Prečo sú bezpečné:** každý používa len vlastný riadok zákazníka, takže žiadna informácia netečie medzi zákazníkmi ani z testovacej sady.

---

## Krok 2 · Rozdeľ raz a testovaciu sadu zamkni

```python
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
CV = StratifiedKFold(5, shuffle=True, random_state=42)
```

**Prečo:** každé rozhodnutie (model, hyperparametre, prah) sa robí 5-násobnou krížovou validáciou **len na trénovacej časti**. Testovacia sada s 1409 zákazníkmi sa použije raz na model, na konci. Keď čokoľvek vyberieš pozeraním na testovaciu sadu, jej skóre prestane byť poctivým odhadom.

**Prečo stratifikovane:** každý fold drží rovnakých 26,5 % odchádzajúcich, takže skóre foldov sú porovnateľné.

---

## Krok 3 · EDA: pozri sa skôr, než modeluješ (`eda` v [`churn.py`](churn.py))

Miera odchodu podľa skupín v najdôležitejších stĺpcoch:
- mesačná zmluva 43 % vs dvojročná 3 %,
- prvých 6 mesiacov 53 % vs po 4 rokoch 10 %,
- optika 42 % vs DSL 19 %.

**Prečo na tom záleží:** tri stĺpce vysvetľujú väčšinu príbehu. Nastaví to očakávania (model sa naučí hlavne toto) a naznačí ľudský baseline.

---

## Krok 4 · Dva baseliny pred akýmkoľvek modelom

- **Naivný:** každý dostane trénovaciu mieru odchodu. Jeho PR-AUC sa rovná podielu odchádzajúcich, **0,265**. To je dno (projekt 06: PR-AUC náhodného modelu = základná miera).
- **Pravidlo:** „mesačná zmluva = odíde“, čo by človek napísal ako prvé. PR-AUC **0,406**.

**Prečo oba:** model je zaujímavý, len keď porazí jednoduché pravidlo. Inak si postavil zložitosť nadarmo.

---

## Krok 5 · Celé predspracovanie daj do `Pipeline`

```python
def preprocess():
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])

def pipe(model):
    return Pipeline([("prep", preprocess()), ("model", model)])
```

**Ako to funguje:** číselné stĺpce dostanú doplnený medián a štandardizujú sa; textové kategórie sa stanú one-hot stĺpcami. Pipeline zlepí predspracovanie a model do jedného objektu s `fit` a `predict_proba`.

**Prečo na tom záleží:** medián, štatistiky škálovania aj zoznam kategórií sa *učia z dát*. Keď ich spočítaš na všetkých dátach pred rozdelením, validačný fold ovplyvnil tréning: malý leak. V pipeline ich krížová validácia automaticky preučí na trénovacej časti každého foldu.

**Prečo `handle_unknown="ignore"`:** kategória, ktorú tréning nevidel (napr. nový spôsob platby v API), sa stane samými nulami namiesto pádu.

---

## Krok 6 · Porovnaj modely krížovou validáciou

Logistická regresia, random forest a LightGBM idú všetky cez `pipe(...)` a hodnotia sa 5-násobným CV PR-AUC (priemer ± smerodajná odchýlka), potom raz na testovacej sade.

**Prečo začať logistickou regresiou:** je rýchla, interpretovateľná a pri tabuľkových dátach silná referencia. Tu má 0,633 test PR-AUC, blízko všetkého ostatného.

**Prečo smerodajná odchýlka:** pri ±0,02 medzi foldmi je rozdiel 0,01 medzi modelmi šum. Lekcia o √n z projektu 06 v praxi.

---

## Krok 7 · Vylaď LightGBM cez Optunu

```python
def objective(trial):
    params = {"n_estimators": trial.suggest_int(...), "learning_rate": trial.suggest_float(..., log=True),
              "num_leaves": ..., "reg_lambda": ..., ...}
    return priemerné CV PR-AUC pre pipe(lgbm(**params))
study.optimize(objective, n_trials=40)
```

**Ako to funguje:** Optuna navrhuje hyperparametre, z predošlých pokusov sa učí, ktoré oblasti sú sľubné, a navrhuje lepšie (múdrejšie než grid search). Cieľ je CV PR-AUC len na trénovacích dátach.

**Čo našla:** 550 malinkých stromov so **4 listami**, nízky learning rate a silnú L2 regularizáciu. **Čo to hovorí:** dáta majú málo zložitej štruktúry; signál sú hlavne pár sčítateľných efektov. Preto je aj logistická regresia takmer rovnako dobrá. Najlepšie test PR-AUC: **0,664**.

---

## Krok 8 · Sprav leakage zámerne a chyť ho (`leakage`)

```python
# ZLE: najprv oversampling, potom split
Xo, yo = oversample(Xtr, ytr, rng)          # kopíruj odchádzajúcich, kým triedy nie sú rovnaké
for tr, va in CV.split(Xo, yo): ...

# DOBRE: najprv split, oversampling len na trénovacej časti každého foldu
for tr, va in CV.split(Xtr, ytr):
    Xf, yf = oversample(Xtr.iloc[tr], ytr.iloc[tr], rng)
    p = make().fit(Xf, yf).predict_proba(Xtr.iloc[va])[:, 1]
```

**Čo sa pokazí:** kopírovanie odchádzajúcich pred CV dá kópie *toho istého* zákazníka aj do trénovacieho, aj do validačného foldu. Model sa potom validuje na riadkoch, ktoré si zapamätal.

**Ako sa to dokáže:** každý riadok sa zahashuje (`hash_pandas_object`); 89 % validačných odchádzajúcich malo presnú kópiu v trénovacom folde. CV PR-AUC: **0,893 s leakom vs 0,638 správne**; skutočný test hovorí 0,655.

**Ponaučenie:** podozrivo dobrý výsledok je dôvod hľadať leak. Všetko, čo sa učí z dát (imputer, scaler, oversampling), patrí *dovnútra* foldu. A správny oversampling aj tak nepomohol (0,638 vs 0,640 bez neho).

---

## Krok 9 · Prah vyber podľa peňazí, nie 0,5

```python
OFFER, LOST, SAVE_RATE = 100, 500, 0.5
THEORY_THRESHOLD = OFFER / (SAVE_RATE * LOST)       # 0,40

def best_threshold(y, p):
    grid = np.arange(0.01, 1.0, 0.01)
    costs = [cost_per_customer(y, p >= t) for t in grid]
    return grid[np.argmin(costs)]
```

**Matica nákladov:** ponuka stojí 100 €; stratený zákazník 500 €; ponuka zachráni polovicu tých, čo by odišli. Ponuka sa oplatí, keď `100 + p·250 < p·500`, čiže **p > 0,40**.

**Prečo aj empiricky hľadať:** pravidlo 0,40 platí, len ak je `p` skutočná pravdepodobnosť. Skript preto vyberie prah s najnižšími nákladmi na **out-of-fold predikciách trénovacej sady** (`cross_val_predict`), nikdy na testovacej. Výsledok **0,37**, blízko teórie.

**Výsledok na teste:** 118,2 € na zákazníka vs 132,7 € „nikdy neponúkni“ a 166,4 € „ponúkni všetkým“. Krivka nákladov je medzi 0,3 a 0,5 plochá, takže vyhnúť sa extrémnemu prahu je dôležitejšie než presná hodnota.

---

## Krok 10 · Skontroluj kalibráciu

Model je kalibrovaný, keď „30 % riziko“ naozaj znamená, že odíde 30 zo 100 takých zákazníkov. `calibration_curve` rozdelí predikcie do košov a porovná priemernú predikciu so skutočnou mierou; **Brier skóre** je stredná kvadratická chyba pravdepodobností.

**Zistenia:**
- LightGBM aj logistická regresia sa trénujú na log-loss, takže sú už kalibrované (priemerná predikcia 0,266 vs skutočnosť 0,265). Izotonická kalibrácia (`CalibratedClassifierCV`) tu nič nezmení; finálny model ju má aj tak ako poistku pre zmenu dát.
- **`class_weight="balanced"`** váži odchádzajúcich ~2,8× viac. Každá pravdepodobnosť stúpne (priemer 0,41), pravidlo 0,40 pošle 674 ponúk namiesto 420 a náklady stúpnu na 124,1 €.

**Ponaučenie:** triky na vyváženie nepridávajú informáciu, len skreslia pravdepodobnosti. Ak vyvažuješ, musíš prah vybrať znova; lepšie nevyvažuj a prah vyber z nákladov.

---

## Krok 11 · Vysvetli model cez SHAP (`explain`)

- **Vstavaná dôležitosť LightGBM** (počet splitov) uprednostňuje spojité stĺpce (`tenure`, platby), lebo ponúkajú veľa miest na split, aj pre malinké zisky.
- **SHAP** meria, o koľko každý príznak pohne každú predikciu (v log-odds). Na vrchu: mesačná zmluva, tenure, chýbajúce online zabezpečenie.
- Beeswarm graf ukazuje aj **smer**: krátka tenure tlačí k odchodu.

**Prečo oboje:** „ako často model použil tento stĺpec“ a „o koľko tento stĺpec zmenil odpoveď“ sú rôzne otázky. SHAP odpovedá na tú, ktorá ťa zvyčajne zaujíma.

---

## Krok 12 · Analýza chýb: koho model nezachytí?

119 z 374 odchádzajúcich nedostalo ponuku. Zachytení sú takmer všetci na mesačnej zmluve s krátkou tenure. Nezachytení vyzerajú ako verní zákazníci (medián tenure 28 mesiacov, veľa na dlhých zmluvách). **Nič v týchto stĺpcoch nehovorí, že sa chystajú odísť**; potreboval by si dáta ako sťažnosti alebo ponuky konkurencie. To hovorí, že ďalšie zlepšenie sú lepšie dáta, nie väčší model.

---

## Krok 13 · Nasaď ho cez FastAPI ([`app.py`](app.py))

```python
class Customer(BaseModel):
    Contract: Literal["Month-to-month", "One year", "Two year"]
    tenure: int = Field(ge=0, le=100)
    ...

@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    X = add_features(pd.DataFrame([customer.model_dump()]))[BUNDLE["columns"]]
    p = float(BUNDLE["pipeline"].predict_proba(X)[0, 1])
    return Prediction(churn_probability=p, send_offer=p >= BUNDLE["threshold"], ...)
```

**Prečo každá časť:**
- **Uložený balík** obsahuje pipeline, prah, poradie stĺpcov a náklady: všetko, čo treba na zopakovanie rozhodnutia.
- **Tá istá funkcia `add_features`** ako pri tréningu, takže príznaky sa počítajú rovnako (klasická chyba v produkcii je počítať ich inak).
- **Typy `Literal`:** preklep ako `"Contract": "monthly"` vráti HTTP 422 namiesto potichu zlej predikcie.
- **[`api_check.py`](api_check.py)** pošle všetkých 1409 testovacích zákazníkov cez endpoint a porovná s offline predikciami: max rozdiel 0,0. To dokazuje, že API servíruje ten istý model.

---

## Krok 14 · Vizuálny príklad, ktorý sedí ([`threshold_gif.py`](threshold_gif.py))

Churn nemá prirodzenú analógiu s autíčkami, preto namiesto kapitoly v Racing Lab GIF ukazuje testovacích zákazníkov ako bodky podľa predpovedanej pravdepodobnosti, prah, ktorý cez ne prechádza, a náklady, ktoré sa menia, až kým nezastane na prahu vybranom z trénovacích dát.

```bash
uv run python data.py && uv run python churn.py && uv run python plot.py
uv run uvicorn app:app --reload
```

---

## Vyskúšaj sám

1. Zmeň `SAVE_RATE` na 0,3. Prepočítaj teoretický prah (`100 / (0,3 · 500)`) a over, že empirický ho nasleduje.
2. Odstráň `charge_change` a `avg_monthly`. Klesne PR-AUC, alebo boli zbytočné?
3. Nafituj `StandardScaler` na celom `X` pred rozdelením a porovnaj CV skóre. Je tento leak tu veľký alebo malý, a prečo?

## Zhrnutie

| Krok | Myšlienka |
|---|---|
| 2 | Zamkni testovaciu sadu; všetky voľby cez CV na trénovacích dátach |
| 4 | Najprv naivný a ľudský baseline |
| 5 | Všetko naučené z dát patrí do `Pipeline` |
| 7 | Ladenie našlo malinké stromy: signál je jednoduchý |
| 8 | Oversampling pred splitom = leak; 0,89 vyzeralo skvele a bola to lož |
| 9 | Prah = biznisové rozhodnutie odvodené z nákladov |
| 10 | Vyvažovanie skresľuje pravdepodobnosti |
| 11 | SHAP ≠ dôležitosť podľa počtu splitov |
| 13 | Servíruj tie isté príznaky aj model, validuj vstupy, over celý reťazec |
