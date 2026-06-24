# TakeMeter — Discourse Quality Classifier for r/TrueFilm

## Community Choice

I chose r/TrueFilm, a Reddit community dedicated to serious film discussion. Posts range from structured analytical essays to casual emotional reactions after a first watch, with a large middle ground of posts that state opinions without fully arguing them. This variation in discourse quality and intent makes it well-suited for a classification task - the distinctions between critique, opinion, and reaction are ones the community itself cares about and actively polices. Unlike a general film subreddit, r/TrueFilm attracts users who self-select for substantive discussion, which means the Critique/Opinion boundary is genuinely contested and interesting rather than trivially obvious.

---

## Label Taxonomy

### critique
A structured argument about a film with a specific claim supported by verifiable evidence — scene references, structural choices, comparisons to other films. Personal framing ("I think", "I love") does not disqualify a post from being Critique if removing that framing still leaves a structured argument.

**Example 1:**
> "The ending of In a Lonely Place succeeds where Suspicion fails because it only exonerates Dix of murder without whitewashing his volatility and violence. The film recognizes that Dix's dangerous impulsivity destroyed any chance they had of a happy marriage regardless of whether he's a murderer — and that forces the audience to reflect on their own tendencies in ways a cleaner ending never could."

**Example 2:**
> "Obsession feels like a story I've already seen before. My issue is that it doesn't fully commit. I turn back to Smile, where we understand why Rose has such an obsession towards her job — it comes from witnessing her mother's death. With Obsession we literally do not know Bear at all. If we compare Nikki's possession state to the sunken place in Get Out, Get Out pushes the narrative further to address racism and eugenics. Obsession on the other hand is a lot more linear and fails to utilize its environment."

---

### opinion
A clear evaluative stance without supporting arguments. The post is asserting, not arguing. If you remove the personal framing and nothing is left but a stance, it is Opinion.

**Example 1:**
> "Why do all teen-oriented blockbusters feel the same? Maze Runner, Hunger Games, Divergent — I would challenge you to tell any of them apart. I really hate how studios aim things for teenagers because I have yet to see a teen blockbuster that wasn't completely lame. Hollywood has absolutely no idea how to write characters in the 13-18 age group."

**Example 2:**
> "My personal view of why she does what she does is a mixture of self destruction, daddy issues, and a Lolita-complex. I think she is firstly depressed and part of her depression is manifesting in self destructive tendencies. A lot of young and beautiful girls feel like they are expiring. Each birthday feels like aging out of a prime."

---

### reaction
The post is primarily about what the author felt while watching the film, not what the film is or means. Emotional or impressionistic experience of watching.

**Example 1:**
> "I just finished watching The Skin I Live In and I need to talk about it because it really blew my mind. I wasn't ready for where the story went. What really got to me was the Stockholm syndrome part — the director doesn't show it as weakness. It's about survival. That's the part that really stays with you."

**Example 2:**
> "Disclosure Day — almost great, and that almost really hurts. I went in absolutely buzzing. Spielberg. UFOs. Emily Blunt. The trailers had me convinced this was going to be something special. But the movie just doesn't stick the landing. You leave understanding the message and forgetting the movie. Really wanted more."

---

## Data Collection

**Source:** Posts collected manually by browsing r/TrueFilm on Reddit. Only public posts were used.

**Labeling process:** Each post was labeled by applying a three-question decision rule in order: (1) Is the post primarily about what the author felt watching? If yes → Reaction. (2) If you remove personal framing, is there still a structured claim with specific verifiable evidence? If yes → Critique. (3) Otherwise → Opinion.

**Final label distribution:**

| Label | Count |
|-------|-------|
| critique | 66 |
| opinion | 40 |
| reaction | 35 |
| **Total** | **141** |

**Train / Validation / Test split (70/15/15):**

| Split | Size |
|-------|------|
| Train | 98 |
| Validation | 21 |
| Test | 22 |

**Train label distribution:** Critique 46, Opinion 28, Reaction 24

---

### Difficult-to-Label Examples

**1. "Obsession: Male Passivity"**
This post makes a specific claim (the film critiques introverted rather than extroverted toxic masculinity) and supports it with scene-specific evidence (the phone call scene, Bear's repeated failure to act). However, it uses heavy personal framing ("what I love about Obsession") and ends with "brilliant film." It could be Critique or Opinion.
*Decision:* Labeled **Critique** — removing the personal framing still leaves a structured argument with specific scene evidence.

**2. "Kill Bill is a metaphor for grooming"**
This post has a specific interpretive claim and supporting evidence, but the author explicitly says "just needed to get these thoughts out, valuable or not" and the structure is associative rather than building toward a conclusion.
*Decision:* Labeled **Reaction** — the author signals they are thinking out loud rather than arguing, and the post is more about their experience of rewatching than a structured claim.

**3. "Eyes Wide Shut makes perfect sense once you know the cabdriver's name"**
Casual tone and exclamation points make this feel like Opinion, but the post makes a specific verifiable claim (the film is structured as a journey through the underworld) supported by checkable details — Charon the cabdriver, Milich as Cerberus, the DOG reflection.
*Decision:* Labeled **Critique** — tone is not the label, structure is. Removing the casual framing leaves a real argument.

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` from HuggingFace

**Training setup:** Fine-tuned on Google Colab using a free T4 GPU. Training took approximately 5 minutes.

**Key hyperparameter decisions:**
- **Learning rate:** Increased from the default `2e-5` to `5e-5`. The default learning rate produced a model that predicted Critique for every example across all epochs with no improvement — a sign the model was not learning. Increasing the learning rate to `5e-5` allowed the model to start learning the label boundaries.
- **Epochs:** Increased from the default 3 to 6 to give the model more passes over the small dataset.

---

## Baseline

**Model:** Groq `llama-3.3-70b-versatile` with zero-shot prompting.

**Prompt approach:** The system prompt included all three label definitions as written in planning.md, one example post per label, and an instruction to output only the label name in lowercase. The model was given no task-specific training.

**Prompt used:**
```
You are classifying posts from r/TrueFilm, a Reddit community for serious film discussion.
Assign each post to exactly one of the following categories.

critique: A structured argument about a film with a specific claim supported by verifiable evidence...
opinion: A clear stance or evaluative judgment without supporting arguments...
reaction: The post is primarily about what the author felt while watching the film...

Respond with ONLY the label name, in lowercase.
Valid labels: critique, opinion, reaction
```

---

## Evaluation Report

### Overall Accuracy

| Model | Accuracy |
|-------|----------|
| Baseline (zero-shot) | 0.727 |
| Fine-tuned DistilBERT | 0.545 |

### Per-Class Metrics

**Baseline:**

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| critique | 0.71 | 1.00 | 0.83 | 10 |
| opinion | 1.00 | 0.33 | 0.50 | 6 |
| reaction | 0.67 | 0.67 | 0.67 | 6 |
| **macro avg** | 0.79 | 0.67 | 0.67 | 22 |

**Fine-tuned:**

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| critique | 0.67 | 0.80 | 0.73 | 10 |
| opinion | 0.38 | 0.50 | 0.43 | 6 |
| reaction | 0.50 | 0.17 | 0.25 | 6 |
| **macro avg** | 0.51 | 0.49 | 0.47 | 22 |

---

### Confusion Matrix (Fine-Tuned Model)

| | Predicted Critique | Predicted Opinion | Predicted Reaction |
|---|---|---|---|
| **True Critique** | 8 | 2 | 0 |
| **True Opinion** | 2 | 3 | 1 |
| **True Reaction** | 2 | 3 | 1 |

---

### Wrong Predictions Analysis

The model got 10 of 22 test examples wrong. Two dominant error patterns emerge from examining the failures.

**1. True: Reaction → Predicted: Critique (confidence 0.61)**
A post about how a specific line in No Country for Old Men hit differently on rewatch was predicted as Critique. The post references a specific scene and quote, which the model likely treated as evidence of a structured argument. But the post is about the author's personal experience of rewatching, not a claim about the film's meaning. The model learned that specific scene references signal Critique, without learning that emotional narration can also contain scene references. This is the most common error pattern — 5 of the 10 wrong predictions involve Reaction posts being mislabeled as Critique or Opinion.

**2. True: Reaction → Predicted: Opinion (confidence 0.67)**
A post about watching One Battle After Another twice in two days was predicted as Opinion. The post is clearly a Reaction — the author is processing the experience of rewatching — but it contains evaluative sentences ("I'd love to read the screenplay") that the model read as Opinion signals. The model confused evaluative language with Opinion, missing that Reaction posts frequently contain evaluative moments without that making them Opinion.

**3. True: Critique → Predicted: Opinion (confidence 0.64)**
A post arguing that mediocre films get defended as "fun" (Scary Movie 6) is a genuine Critique making a structured argument, but the conversational tone led the model to predict Opinion. This reveals the model learned that formal or dense language signals Critique, missing that a structured argument can be written conversationally. The model is using tone as a proxy for label when the real signal is argument structure.

---

### Sample Classifications

| Post excerpt | True Label | Predicted Label | Confidence |
|---|---|---|---|
| "I just finished watching The Skin I Live In and I need to talk about it because it really blew my mind..." | reaction | reaction | 0.61 |
| "Why do all teen-oriented blockbusters feel the same? Maze Runner, Hunger Games, Divergent..." | opinion | critique | 0.58 |
| "The ending of In a Lonely Place succeeds where Suspicion fails because it only exonerates Dix of murder..." | critique | critique | 0.74 |
| "Barry Lyndon and the Sublime — Kubrick uses the juxtaposition of natural beauty and human pettiness..." | critique | critique | 0.71 |
| "Obsession is a MUST-SEE theatrically! I literally jumped, laughed and commented watching this with my crowd..." | reaction | opinion | 0.52 |

The third example (In a Lonely Place) is a correctly predicted Critique. The prediction is reasonable because the post makes a specific comparative claim about two film endings and supports it with detailed scene-by-scene reasoning — exactly what the Critique label is designed to capture. The model's relatively high confidence (0.74) reflects that this post has clear structural markers of an argument.

---

### Reflection: What the Model Learned vs. What I Intended

I intended the model to learn the difference between structured argumentation, bare assertion, and emotional narration. What it actually learned is closer to surface-level signals: long posts with film references → Critique, short evaluative posts → Opinion, posts starting with "I just watched" → Reaction.

This is most visible in the Opinion/Reaction confusion. Both labels can contain evaluative language and both can be short - the difference is about *intent and structure*, which is a subtle distinction that requires understanding the whole post. With only 24-28 training examples per label, the model never had enough signal to learn that distinction reliably.

The Critique label worked best because Critique posts on r/TrueFilm tend to be long, densely referenced, and structurally distinctive — the surface features align well with the true label definition. Opinion and Reaction are harder because their surface features overlap significantly.

---

## Spec Reflection

**One way the spec helped:** The warning against vague labels ("good" vs "bad") was the most useful guidance in the entire project. My first instinct was to label posts as high-quality vs low-quality discourse, which would have been unannotatable. The spec pushed me toward grounded, community-specific labels that I could actually apply consistently.

**One way implementation diverged:** The spec assumes 200 balanced examples is achievable. In practice, r/TrueFilm skews heavily toward long-form Critique posts — Reaction posts are rare because the community culture discourages pure reaction posts. I ended up with only 141 examples after rebalancing, which fell below the spec minimum but produced a more honest training set than an artificially padded imbalanced one.

---

## AI Usage

**1. Label stress-testing:** I gave Claude my three label definitions and asked it to generate 8-10 posts sitting at the Critique/Opinion boundary. Working through those generated posts revealed a gap in my decision rule — I had not specified that subjective *framing* of evidence doesn't disqualify a post from being Critique. I added the following to my rule: "Evidence is specific enough if it references a verifiable element of the film. Subjective framing of that evidence doesn't disqualify it." This change affected how I labeled several real posts during annotation.

**2. Annotation assistance:** I used Claude to pre-label approximately half of my 141 examples by providing my label definitions and asking for one label per post with a brief reason. I reviewed and corrected every pre-labeled example before adding it to my CSV — I overrode Claude's suggestions in roughly 15-20% of cases, mostly where Claude labeled as Opinion posts I judged to be Critique based on specific evidence I recognized from knowing the films discussed.
