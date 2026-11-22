# Slide 1 — Introduction: Which Exposure Matters?

This guide explains the first slide for someone who is new to generative models and diffusion. It connects the title, the noise illustration and the research question. The final section is a short script you can present to the audience.

## 1. What is the main idea?

Our project asks a simple but subtle question:

> If a model has often seen the individual ingredients of an image, does it also need to see a particular combination of those ingredients to generate that combination successfully?

For example, suppose a model has seen many red objects and many squares. Does that automatically mean it will generate a red square correctly? Not necessarily. It may have seen many red circles and many blue squares but relatively few red squares. The number of times it saw the **individual ingredients** and the number of times it saw the **combination** are different things.

We propose studying this distinction in a controlled synthetic environment using a conditional diffusion model. Our work has built substantial measurement and pilot infrastructure, but the direct experiment that varies joint exposure at controlled marginals remains to be completed.

## 2. What does the title “Which Exposure Matters?” mean?

**Exposure** describes what occurs in the model's training data and how often the training procedure presents it. In this project we distinguish:

- **Marginal exposure:** how often an individual event or ingredient occurs, such as a scene containing a red object.
- **Joint exposure:** how often a specified complete combination occurs, such as a red object that is itself a square.
- **Model success:** whether an image generated after training satisfies the requested rule. This is an output measure, not training exposure.

The research question is not just “does more data help?” It is more specific: if we hold relevant individual-event exposures controlled, does changing full-rule exposure predict a difference in rule satisfaction? This requires carefully designed datasets and reliable evaluation; the title alone does not imply an answer.

### A simple 100-image example

Imagine two illustrative training distributions. In each, 50 of 100 examples contain event A and 50 contain event B. In the first distribution the two events occur together in only 10 examples; in the second they occur together in 40. The two individual frequencies are the same, while the joint frequency differs. Slide 4 shows complete probability tables for this example.

If A means “red exists” and B means “a square exists” in a multi-object scene, both can be true for different objects. A bound **red-square rule** additionally requires the red color and square shape on the same object. We must measure that bound event, not assume that two separate existential statements imply the combination.

The numerical example illustrates a possible data contrast. It does not say that a model trained on either distribution has been tested or that one necessarily performs better.

## 3. What is a generative model?

A **generative model** is a model designed to produce samples resembling a learned data distribution. In image generation, its output is an image instead of a single category label.

A **conditional** generative model also receives information describing the desired output. For instance, the condition could be a color label or a structured rule. A condition is a request, not a guarantee: the produced image must still be checked against it.

In our reported four-condition pilot the model is conditioned on a requested target color, red or blue. It is not a general natural-language text-to-image system, and it is not yet the planned model experiment with multi-predicate exposure mixtures.

## 4. What is diffusion, in plain language?

A diffusion model learns denoising-related behavior across a sequence of noise levels. One way to understand it is to separate **training** from **generation**.

### During training

1. Take a clean training image.
2. Select a diffusion timestep, which corresponds to a noise level.
3. Combine the clean image with sampled Gaussian noise according to a schedule.
4. Give the noisy image, the timestep and the condition to the neural network.
5. Compare the network's prediction with a known training target and update the model.

In our pilot, the network predicts a **velocity target** made from the clean image and the sampled noise. It is not literally predicting the speed of a moving shape. Slide 6 presents the noising and velocity equations; slide 8 explains the optimizer and EMA.

### During generation

1. Start with a newly sampled noise tensor.
2. Provide the desired condition to the trained denoiser.
3. Use the network's prediction in a reverse-sampling step.
4. Repeat through the configured reverse schedule to produce an image.
5. Evaluate whether the output actually satisfies the requested condition.

The reported pilot uses 1,000 reverse diffusion steps for its fixed sampling protocol. The sampler does not simply look up an existing training picture, though a model trained on a small fixed dataset can still memorize or closely resemble examples; that must be investigated rather than assumed away.

## 5. Explain the three pictures on the slide

The pictures are labeled **Noise**, **Less noise** and **Clean image**. They provide an intuitive direction: a noisy starting point can be transformed through intermediate states toward an image-like output.

The captions on the slide matter. These are **illustrative linear blends** of a clean synthetic render and fixed noise. They are not actual model-generated intermediate samples, and they do not depict the exact DDPM noise schedule. The rightmost clean shape picture is an illustration of the target concept, not evidence that this specific image was generated by the trained model.

A useful presentation sentence is: “This sequence is a visual analogy for the denoising direction; the model's actual sampling process uses a specified schedule and learned predictions.”

## 6. Why is composition difficult?

A generated image can contain the right ingredients but combine them incorrectly. For a requested **red circle left of a blue square**, several distinct errors are possible:

1. The red object is a square and the blue object is a circle: the colors and shapes exist but are bound to the wrong objects.
2. The right objects appear, but the red circle is to the right of the blue square: the relation is wrong.
3. Only one of the requested objects appears.
4. The objects merge, overlap ambiguously or cannot be recognized reliably.

A correct outcome must satisfy the full rule under the stated object and relation semantics. For our prototype rule checker, “left of” means a strict comparison of object-center x-coordinates; it does not require the objects to be on exactly the same horizontal line. “Distinct objects” means two different objects, not necessarily different colors.

The four-cell feasibility pilot's primary review question is narrower: whether a requested-color circle is present. It does not test the entire example rule above.

## 7. Why use synthetic scenes?

The team built a renderer for small scenes of colored shapes. Because the renderer records attributes and positions, we can define explicit predicates and estimate how frequently they hold. We can also construct different generating conditions and use optimization to choose mixtures of those conditions.

This setting has several strengths:

- The data-generation process and the meaning of a rule are explicit.
- We can count individual events separately from correctly bound complete rules.
- We can attempt controlled changes to training distributions.
- The smaller model and images make preliminary experiments practical on limited hardware.

It also has limits: flat 64 × 64 shapes are simpler than natural images, and synthetic-scene findings do not automatically generalize to large real-world text-to-image models. Some prototype relation definitions and template combinations require further validation.

## 8. How do the three course areas connect?

### Digital Image Processing (DIP)

We render and inspect images, represent objects by pixels and annotations, add Gaussian noise, examine generated outputs, and develop image-based checkers. These are image construction, transformation and evaluation tasks.

### Computational Optimization (CoMA)

We choose training-mixture weights subject to explicit constraints, study convex objectives such as minimum KL divergence, and explore a separate nonlinear max–min design. We also train neural-network parameters using an optimizer. Optimizing the **data mixture** and optimizing the **model weights** are different problems.

### Computer Networks (CN)

Prototype infrastructure streams PNG scenes and JSON annotations over framed TCP through SSH forwarding, transfers compressed checkpoints with resume and SHA-256 verification, and coordinates independent inference jobs across workers. The networking checks support the research workflow; they do not prove a GPU-training speedup. Slide 11 explains the measured network runs and their scope.

## 9. What has the project done, and what is the proposed next study?

The team has built a scene renderer, exposure measurement and mixture-design tools, a conditional U-Net training pipeline, pilot evaluation workflows and networking prototypes. A documented exposure-measurement run collected 5.7 million accepted synthetic scenes across 57 conditions. The separate four-condition pilot used 200 fixed training scenes per condition and evaluated a first block of 64 generated images.

Those results demonstrate concrete engineering work and provide diagnostics. They do **not** establish the main hypothesis because the pilot changed object count and color balance rather than running the proposed controlled joint-exposure intervention. The next study must validate the mixture contrast and evaluation, train comparably configured models with independent dataset and training seeds, and measure full-rule satisfaction.

## 10. Likely evaluator questions

**Is diffusion the same as adding noise?** Adding noise is the known forward training process. The learned model supports the reverse sampling process that generates an image from noise.

**Does a clean-looking image prove success?** No. It must be evaluated against the actual request. A visually plausible image may have the wrong color, binding or spatial relationship.

**What is the difference between marginal and joint exposure?** Marginal exposure counts an individual event; joint exposure counts the specified combination. In multi-object scenes, same-object binding must be checked explicitly.

**Are the pictures on slide 1 actual model outputs?** No. They are illustrative blends intended to explain the direction from noise toward an image.

**Has the title question been answered?** Not yet. We have prepared tools and run a separate feasibility pilot; the direct controlled joint-exposure study remains to be performed.

## What to tell the audience

> “Our project is called ‘Which Exposure Matters?’ We ask whether a diffusion model needs to see a specific combination, such as a red square, even when it has already seen many red objects and many squares separately. A diffusion model learns from noisy training images and generates by gradually transforming new noise into an image. The pictures here are an illustration of that process. We built a controlled world of colored shapes and tools to measure individual and combined exposures; the direct exposure experiment is our next step.”
