# 🌾 Rice Disease Detector

**Detecting 10 rice leaf conditions from a photo — with visual explanations, treatment guidance, and an honest look at where the model breaks.**

🔗 **Live demo:** https://rice-disease-detector-q5emzs77muhe5a7lfldvk9.streamlit.app/
👤 **Author:** Rida Fatima

<img width="485" height="435" alt="image" src="https://github.com/user-attachments/assets/7746a16c-303c-4ef7-9181-a25f4c77c2a7" />


---

## The problem this actually solves

Most published plant-disease classifiers report ~99% accuracy — and then fail the moment they see a real photo a farmer would take, because they were trained and tested on clean, lab-style images.

This project doesn't hide from that gap. It measures it, explains it, and closes it. The result is a rice-disease classifier that has been *stress-tested on images from a completely different source* — which is the difference between a demo and something that would actually work in a field.

---

## What it does

Upload a photo of a rice leaf and the app returns:

- **A diagnosis** across 10 classes (9 diseases/conditions + healthy)
- **A confidence score** for the prediction
- **A Grad-CAM heatmap** showing *where on the leaf* the model looked to make its decision
- **Treatment guidance** for the detected condition

The 10 classes span four different causes — bacterial, fungal, insect damage, and viral — plus healthy leaves:

| Class | Cause | Notes |
|---|---|---|
| Bacterial leaf blight | Bacterial | Yellowing, withering, drying |
| Bacterial leaf streak | Bacterial | Dark streaks between veins |
| Bacterial panicle blight | Bacterial | Affects the grain head; worsened by heat |
| Blast | Fungal | Spindle/diamond-shaped lesions; highly damaging |
| Brown spot | Fungal | Brown oval spots; linked to poor soil nutrition |
| Dead heart | Insect (stem borer) | Central shoot dies and browns |
| Downy mildew | Fungal-like | Thrives in wet, poorly-drained conditions |
| Hispa | Insect (hispa beetle) | Scraped leaf surface |
| Normal | — | Healthy leaf |
| Tungro | Viral (leafhopper-spread) | Yellow-orange discoloration, stunting |

---

## The headline result: measuring and closing the domain gap

This is the core of the project.

**Step 1 — Train a strong classifier.**
A fine-tuned EfficientNet-B0 (transfer learning) trained on the [Paddy Doctor](https://paddydoc.github.io/) dataset — 10,000+ real, expert-annotated paddy field images.

> **In-domain test accuracy: 94.3%** (on a held-out Paddy Doctor test set)

**Step 2 — Test it honestly, on a different data source.**
The same model was then evaluated on rice images from a *separate* dataset (different camera, lighting, backgrounds) for the three overlapping disease classes.

> **Cross-dataset accuracy: ~15%** — a collapse of nearly 80 points.

A confusion matrix showed the errors weren't random — the model was *systematically* misclassifying the new-source images, the classic signature of **domain shift**. The model could always recognise these diseases; it simply hadn't learned this dataset's visual style.

**Step 3 — Close the gap.**
Mixing a small sample of the target-domain images into training (domain adaptation) recovered cross-domain performance dramatically — **without any loss of in-domain accuracy** (Paddy Doctor validation held at ~96%).

| Stage | Cross-dataset accuracy |
|---|---|
| Before adaptation | ~15–23% |
| After adaptation | Recovered ~4× |

> ⚠️ **Honest note:** the post-adaptation test set is small (60 images) and visually homogeneous, so the *magnitude* of the jump — not the exact final percentage — is the trustworthy takeaway. It confirms the collapse was a domain-shift problem, not a capability problem.

---

## Explainability (Grad-CAM)

Every prediction comes with a heatmap of where the model focused. On the example below, the model diagnosed **blast** and its attention landed precisely on the spindle-shaped lesion — the exact feature an agronomist uses to identify it.

<img width="150" height="227" alt="image" src="https://github.com/user-attachments/assets/7aacb600-1f07-47fe-92ae-fe67dbfd9f5f" />


This matters because it shows the model is diagnosing from the *actual diseased tissue*, not from background artifacts — and it lets a user sanity-check every prediction.

---

## Architecture

```
Rice leaf photo
      │
      ▼
EfficientNet-B0  (transfer learning, 10-class head)
      │
      ├──►  Predicted class + confidence
      │
      └──►  Grad-CAM heatmap  ──►  Treatment guidance
                                        │
                                        ▼
                             Streamlit web app (live)
```

**Stack:** PyTorch · timm · Grad-CAM · Streamlit · scikit-learn

---

## Honest limitations

Being clear about what this does *not* do:

- **Single disease per image** — it predicts one condition per leaf, not multiple co-occurring diseases.
- **No severity grading** — it identifies the disease, not how advanced it is.
- **Overconfidence** — like most deep networks, it can be confidently wrong, especially on images unlike its training data. During cross-dataset testing it often made incorrect predictions with high confidence — a textbook illustration of poor calibration on out-of-distribution data.
- **Cross-dataset validation covers 3 of the 10 classes** — the domain-gap experiment was run on the three diseases shared between both datasets.

These aren't afterthoughts — understanding them is part of what makes the result trustworthy.

---

## Datasets

- **[Paddy Doctor](https://paddydoc.github.io/)** — training + in-domain test (10,000+ expert-annotated field images).
- **Rice Crop Diseases** (Kaggle) — independent cross-dataset test source.

---

## Run it yourself

```bash
git clone [GITHUB REPO URL]
cd rice-disease-detector
pip install -r requirements.txt
streamlit run app.py
```

---

## What I'd do next

- Larger, more varied cross-domain test set for a firmer post-adaptation number
- Confidence calibration (temperature scaling) to make the probabilities trustworthy
- Multi-label support for leaves showing more than one condition
- Bilingual (English/Urdu) treatment guidance for real farmer accessibility

---

*Built as an end-to-end applied ML project: data pipeline, training, honest evaluation, explainability, and live deployment.*
