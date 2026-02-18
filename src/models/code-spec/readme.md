# Speculative decoding

## so what is standard vs speculative decoding ?

* **standard** is nothing but generate one forward pass = 1 token with one main model
* **sepculative** will have two models draft & main model where you generate say 5 tokens in draft and then pass it to main model generate K tokens

> one forward pass = K tokens so you'll see inference speed

## what is the draft model ?

Draft model is student model which is learned from the teacher (main model)
where it is tiny than main, say for example

### here in the exmaple we using

| Role | Model Size |
| :--- | :--- |
| **Teacher** | Qwen1.5B |
| **Student** | Qwen0.5B |

infering the student model is very less compute compare to the matmul all weights in the
main model.

## Efficiency & Process

so we pass to student 5 tokens it process paralllely
to the main model which forward 5 at once rather than 1 at once
this help us distrubute the GPU compute equally rather than wasting utilization of GPU

before returning the response target model perform acceptance rate
to the prev generated tokens


## Results

by adding speculative I can able to increase 4 tokens faster 

Standard single pass decoding:

⚡ Speed: 44.99 tokens/second


with Sepculative:

prompt : write me simple transformer in torch
⚡ Speed: 49.07 tokens/second

