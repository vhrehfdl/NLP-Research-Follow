논문 URL : [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

## 논문 Background

---

[참고 파일]

1) GPT1 : [https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

2) GPT2 : [https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 

3) [https://jalammar.github.io/how-gpt3-works-visualizations-animations/](https://jalammar.github.io/how-gpt3-works-visualizations-animations/)

4) [https://www.youtube.com/watch?v=xNdp3_Zrr8Q&list=PLIV_C_smLSlXKIoH2A93cCKHdBGsD-hJC&index=13](https://www.youtube.com/watch?v=xNdp3_Zrr8Q&list=PLIV_C_smLSlXKIoH2A93cCKHdBGsD-hJC&index=13)

5) [https://www.youtube.com/watch?v=2uGaXv_ds-k&list=PLIV_C_smLSlXKIoH2A93cCKHdBGsD-hJC&index=15](https://www.youtube.com/watch?v=2uGaXv_ds-k&list=PLIV_C_smLSlXKIoH2A93cCKHdBGsD-hJC&index=15)

[설명]

GPT3 논문입니다. 아무래도 큰 줄기가 되는 논문들을 보는것이 좋겠다는 생각하에 계속 고민하다가 선택하게 되었습니다.

## 토론

---

Q1 (정훈). GPT-3를 실서비스에 사용할 수 있을까요? GPT-3는 특정 도메인에 맞춰서 특화시킨 모델을 현재는 만들 수 없습니다. 그렇다고 이 모델을 공개할 수 있을까요? 굉장히 큰 용량의 모델을 다운 받아서 Fine tuning 시킨 다는 것 또한 모든 기업에서 가능하지는 않을 것 같습니다. 

그렇다면 다른 기업에서는 생성 문제를 풀 때 어떤 방법을 선택해야 할까요? (뭔가 정답을 바라는 질문은 아니고 그냥 한번 여러분의 상황이 되었을 때를 가정해보면 재미있을 것 같아 올려봅니다.)

A1 (소연). 어렵네요ㅠㅠ 현실적으로 가장 먼저 해봄직한 방안은, inference가 가능한 형태로 주어진 높른 성능의 모델을 teacher모델로 두고, 제 상황에서 학습 가능한 모델을 student로 둬서 distillation 하는 방법입니다. 근데 GPT3도 유료화 계획이라하고, 우수한 모델 또는 특정 모델을 개발한 회사들이  학습시킨 모델을 intellctual property로 두는 경향도 점점 생겨서,  이런식으로 distillation 하는 것이 model stealing 처럼 간주될 것 같기도 하네요. 

A1 (지성). 가장 고민되는 부분인거 같습니다 ㅠ PLM에 투자된 비용을 생각하면 애저에서 독점적으로 사용하게끔 하는것도 이해가 됩니다만.. 설사 공개를 하더라도 fine turning 조차도 힘든 경우가 많으니 @@ 참 어려운 부분인거 같습니다. 

그런면에서 저는 강화학습을 접목하는 방향을 고민했던거 같아요. (챗봇에 한정이긴 하지만)
데이터의 한계나, 특정 도메인에 맞춰 특화 시키는 부분의 고민을 어느정도 해결할수 있을거 같다는 생각입니다. (물론.. state와 reward를 구상하는것 또한 큰 문제이고, bias 등의 윤리적인 문제가 있긴하지만요 @@)

Q2 (정훈). 예를 들어, 의료 도메인 데이터를 사용해 GPT로 학습(사전 훈련 모델 사용X) 시킨 것과 Bi-LSTM 모델롤 학습시킨 것을 비교해볼때 대부분 GPT가 성능이 좋은 결과를 보았습니다. (성능이 좋다는 것은 일단 완전한 문장을 만들어낸다는 뜻입니다.) 모델의 구조 차이가 그렇게 큰 걸까요?

A2 (소연). 성능 비교에서 데이터의 양과 질을 제하고서 모델 측면에서는 두가지 요소로 많이 결정된다고 생각합니다. 1) 파라미터 갯수 차이, 2) 동일 파라미터 기준 어떻게 구조가 짜여졌는지.

GPT와 bi-LSTM의 모델 규모/ 파라미터 비교가 많이 날 것 같은데, 모델 파라미터가 얼만큼 학습할 수 있느냐의 capacity 관점이면 당연히 큰 파라미터 가진 모델이 더 높은 성능을 띌 가능성이 높다고 봅니다.

동일 파라미터 기준 모델 구조를 어떻게 짰느냐에 따라 성능 자체가 많이 좌우되고, 그게 deep learning computer vision 연구에서 더 적은 파라미터로 efficient하게 모델 구조를 짤것이냐가 이슈였던 이유라고 생각합니다. 당장, 간단한 conv/ FC 블락 몇개 쌓고 돌려도 성능이 꽤 차이 나기도 하구요.(최적 hyperparameter 영향도 있을거고) 같은 성질의? 레이어를 두고도 어떻게 쌓으냐가 영향을 미치는데, 하물며 GPT와 bi-LSTM 의 고안 철학? 이 다르게 짷여진 만큼 그 영향이 더 클 것 같습니다.(lstm이 gradient vanishing / efficient parameter sharing 등을 해결하려 했고, transformer는 lstm의 작동을 attention 을 적용해서 풀고자한걸로 이해해서요 저는!)

Q3 (소연). 제가 주로 봤던 Few-shot(one-shot 포함) task는 주로 metric distance 또는 optimization based meta learning(inner loop/ outer loops을 이용한 bi-optimization)으로 푸는 문제였습니다. 후자의 경우 adaption을 위한 inner loop에서의 gradient update가 사용되고, 당연히 필요한 과정이라고 생각했습니다. 그리고 본 논문의 Page.4 의 하단 주석을 보면 meta learning과 few shot task에 대한 구분을 해두었습니다.

제가 신기한 부분은 Figure 2.1을 보면 zero-shot/ one-shot/ few-shot은 inference에서 gradient update 없이 데이터셋만 몇개 예제로 주고, 제가 원하는 테스크(e.g, translation) 에 맞춤으로 작동하길 기대하는 것으로 이해했습니다. 

vision task로 예를 들자면 만화로 그린 동물을 분류하도록 학습된 classifier가, 추가 학습없이 몇 개의 수채화로 그려진 동물 그림만보고 향후 분류해야한다는 것으로 이해했습니다. 이게 어떻게 가능한 것인가요..? NLP task가 generation task라 가능한 것인가요?

A3 (지성). LM(language model)을 만들면서 '문장의 구조'를 embedding을 통해 표현한 특징 때문이지 않을까 생각합니다 ! 예를들면 책, 글을 많이 읽으면 문장이나, 어휘를 구사하는 능력이 좋아진다고 합니다. 문장과 논리구조가 잘짜여진 글을 통해서 그 문장 구성의 방식을 배우는 거죠 물론 특정 주제에 한정된 글을 읽었을수도 있지만 일반적인 문장을 구사하는 능력은 늘수 있는것 처럼요. 

많은 데이터(다양한 문장, 언어)로 이러한 언어 구조적 특징을 학습했기에 zero-shot/ one-shot/ few-shot 가능하게 되었지 않을까요 ?!  (저가 이해한 바로는 그런거 같습니다.. 다른분들 의견은 어떠신가요 !?) 

A3 (지은). 가장 마지막 부분만을 채우는 형식이라서, NLP라서 가능한 태스크이지 않나 싶습니다. 즉, 하나의 문장 중 마지막 단의 단어만을 제외한 나머지가 전부 given인 상태이고, 기존에 pre-train된 지식이 있기 때문에, 이전 word에 대한 probability가 전부 주어진 상태라서.. 마지막 단에 나올 단어들에 대한 candidate set도 만들 수 있구요. 

이게 Vision task에도 적용이 가능한 방식인지는 모르겠습니다. 비전쪽에도 one-shot, zero-shot 등이 있다는 건 알고 있는데, 데이터를 모델이 인식하는 과정이 좀 다른 것 같아서요. (NLP는 conditional probability와 joint probability를 조합해서 하지만, 비전쪽은 이미지의 shape을 인식하는 edge 체크가 중요할거라는 생각이 들어서요.. 혹시 틀린게 있다면 알려주세요!) 관련 논문들은 아직 안 읽어봐서;; 비슷한 논조의 내용이 논문에도 있기는 합니다. inference 단에서 모델이 '새로 학습'을 하는건지, 아니면 기존에 pre-trained knowledge만으로 추측을 하는건지 알 수 없다-는 내용이 있었죠. 

Q4 (소연). 6.2의 bias, fairness& representation에서 다뤄지듯 모델 자체가 편향된 결과를 낸다는건 문제이지만 이게 과연 모델의 문제라고 볼 수 있을까요? Data driven으로 학습되는 딥러닝 특성상 이부분은 학습시 사용된 데이터 자체의 문제가 더 큰 것 같은데( e.g, 인터넷 크롤링을 한다면 전체적인 bias가 낀 사회 양상이 반영된 데이터가 모여질 확률이 높으니까요..), 데이터 단에서 처리하는 해결법 외에 모델의 학습에서 bias나 fairness를 해결할 수 있는 방안은 어떤 것이 있을까요? 혹은 일반적 모델을 이용해 데이터 처리에 도움이 되도록 하거나요..?

A4 (선민). LM의 잠재 표현(latent representation)에 대한 임베딩과 sentiment prediction-derived 정규화를 통해 비교가능한 수준의 perplexity와 semantic similarity를 유지하면서 fairness metircs를 향상을 제안한 연구가 있다고 합니다. sensitive attribute values (e.g. country names, occupations, person names)의 집합을 고려하여, input sequence에 sensitive token을 사용했습니다.(ex 남자 지칭의 경우 a, 여자의 경우 a') 

(** [https://arxiv.org/abs/1911.03064](https://arxiv.org/abs/1911.03064))

Q4 (민상). GPT-3의 성능이 대단한 건 사실이지만, 파라미터의 숫자나 데이터의 양, 연산에 필요한 시간과 에너지 등을 고려하면 효율적이라고 말할 수 있을지는 잘 모르겠어요. GPT-3 같은 모델을 개발하고 학습시키는 데 장벽이 클 것 같네요. 혹시 제한된 cost(시간, 연산력, 에너지, 파라미터 수 등등) 안에서 모델의 효율성을 올리는 법에 대한 연구들은 없을까요?

A4 (정훈).  제가 아는 cost를 줄이는 방법은 대표적으로는 분산 학습이 있습니다. 요즘 pytorch 같은 경우에는 data paraller 같은 방법들이 내장되어 있을 만큼 여러 GPU를 동시 사용해서 학습 시키는 방법이 대중화(?)되어 있습니다. 민상님께서 말씀해주신 시간과 연산력 부분은 GPT 혹은 TPU 밖에 해결할 수 없는 문제라고 생각됩니다. 물론 파라미터 수는 DL 모델의 크기로 결정되지만요.

사실 GPT-3와 같은 거대 모델은 큰 기업단에서나 가능한 것이기 때문에 실제 중소 규모의 회사에서 만들어낼 수 있을지는 저도 고민이였습니다. 하지만, GPT-3의 말도 안되는 파라미터 숫자 때문에 이러한 성능이 나오는 것도 사실이고... 이것을 앞으로 어떻게 경량화 시키면서 성능을 유지시킬 수 있을지는 앞으로 연구의 주요 화두가 되겠죠? ㅎㅎ 

근데 보면 그 많은 데이터를 모델에 담으려면 저정도 파라미터 숫자가 필요하지 않나 싶습니다.

A4 (소연). 본 논문은 높은 성능을 목적으로 개발된거라 연산의 효율성은 고려되지 않은 것 같다 생각해요! 이러한 모델을 제한된 하드웨어 리소스로 학습시키기 위해 모델 경량화의 방법도 있겠고. 이때 발생하는 메모리 문제 중 멀티 gpu로 학습 배치를 키우거나, 모델 자체를 쪼개서 학습 시키는 방법이 있다고 압니다.(Data parallelism, model parallelism) 그 중에서 쉽게 개발하도록 만들어진 기능이 pytorch dataparallel, distributeddataparallel 같은 것들이 있구요. 

A4 (지성). 공감합니다. large scale LM 구성하는 비용적인 문제나, 데이터적인 면에서도 진입장벽이 크다고 생각합니다 ㅠㅠ. 완전 핏하진 않지만 최근 Parallelformers (TUNiB) 와 같은 자원과 시간적인 cost를 줄이고자 하는 접근의 연구 & 개발은 늘고 있는거 같습니다 ! 

(** [https://github.com/tunib-ai/parallelformers?fbclid=IwAR1R4Hilz3-k4V2Uj_V35FN3LzL_K4yKxwbYHNkvcwA4suyLeXx9dOj7WoQ](https://github.com/tunib-ai/parallelformers?fbclid=IwAR1R4Hilz3-k4V2Uj_V35FN3LzL_K4yKxwbYHNkvcwA4suyLeXx9dOj7WoQ))

Q5 (민상). Limitations 섹션에서 GPT-3가 common sense physics에 대한 작업을 잘 하지 못한다고 쓰여 있는데, 다른 모델 중에서는 일반 상식을 학습하는 모델이 있나요? 특별한 목적이 없는 일반 상식을 어떻게 학습할 수 있을지 궁금해지네요.

A5 (선민). 저도 궁금해서 찾아보니 BERT에서 common sense 데이터로 finetuning한 연구가 있네요. 위키피디아에서 데이터를 가져와 CommonesenseQA 데이터셋을 만들었는데,  1개의 question에 대해 5개의 candidate answers를 연결하여 queation-answer pair sequence로 BERT에 적용했다고 합니다.

(** [https://arxiv.org/abs/2008.03945](https://arxiv.org/abs/2008.03945))

Q6 (민상). 이 연구의 주된 contribution을 뭐라고 생각하시나요? 사실 읽는 내내 '전례 없던 규모로 이런 모델을 개발했는데 시험해보니 이 정도 성능을 내더라'는 내용이 계속 나와서 재밌는 연구라는 생각이 안 들었거든요. (Q4에도 썼듯 소수의 기관에서만 디벨롭할 수 있는 연구인 것 같기도 하고...) 개인적으로는 전인미답의 수준에 도달했다, 자연어 모델의 신기원을 열었다는 것 정도인 것 같은데 다른 분들은 어떻게 생각하시나요?

A6 (정훈).  놀라운 성능이라고 생각합니다. 사실 방식 자체는 이전 GPT2와 거의 유사하지만 말도 안되는 규모의 데이터로 말도 안되는 하드웨어를 사용해 학습한거라고 생각합니다. GPT3는 GPT2보다 더 놀라운 성능을 보였고 이는 파라미터 수와 성능이 비례하며 발전할 수 있다는 가능성을 보였다는 점에서 의의가 있을 것 같습니다. 그렇다면 계속해서 하드웨어가 발전하고 데이터가 증가할 경우 파라미터 수가 인간 뉴런 개수만큼 증가할 경우 학습으로 인간과 비슷한 지능을 모방할 수 있지 않을까라는 꽤나 허무맹랑하지만 그런 생각을 할 수도 있지 않을까 싶습니다. ㅎㅎ

A6 (지은). task-agnostic한 모델로 여러 task에 접목을 시키려고 했고, 실제로도 말도 안되게 좋은 결과를 내었다는 게 아닐까 싶습니다. 연구 목적으로 나온 모델이기 때문에 데이터셋도 모델의 크기도 보면 현재로선 말도 안되는 수준이지만, 꾸준히 파라미터를 줄이는 방법에 대해서는 연구가 되고 있기 때문에, 돌리는 비용 역시도 앞으로 줄어갈 것 같구요. 많은 모델들이 그래왔듯이.. ㅎㅎ 

Q7 (지성).  처음 GPT 논문들을 읽으면서 가졌던 의문점 중 하나인데, GPT 가 생성하는 문장이 문맥을 정말 이해하는게 맞을까 라는 생각입니다. 저는 GPT로 생성된 문장들이 '짜여진 대본을 읽는다'는 느낌을 많이 받더라고요 ! 암기 통해 외운 문장(학습)을 통해 다음올 대사를 밷어내는거 처럼 말이죠 (대화가 아닌).
실제로 GPT는 많은 task에서 높은 성능을 보이지만 NLU task에서는 그렇지 못하고 있는걸로 알고있습니다 (단방향성 학습방식 한계) 엄청난 규모의 데이터로 '좋은 성능인 것 처럼  보이고 있는것은 아닐까 ?' 하는 의문입니다.

A7 (정훈). 이 부분은 저도 많이 고민했던 부분입니다. 우선 GPT가 문맥을 이해하는냐를 얘기하려면 사람은 문맥을 어떻게 이해하느냐가 먼저 정의되어야할 것 같습니다. 사람이 문맥을 이해하는 것은 이전의 문장들을 기반으로 다음 문장이 논리적으로 맞는 부분인지 체크하느냐?가 아닐까요. 

그렇다면 GPT 또한 어느 정도는 문맥을 이해한다고 말을 할 수 있을 것 같습니다. 학습 방식 자체가 이전 문장과 다음 문장을 예측하는 방법이기 때문에 이 알고리즘 구조에 의하면 문맥을 이해한다고 말할 수 있을 것 같습니다.

다만 문제는 GPT가 추론이 가능한가?인 것 같습니다. 현재 모든 인공지능 모델이 추론을 불가능한다고 여겨집니다. 그렇다면 추론을 통해 확장 가능한 문맥이해는 현재까지는 안되는것으로 보는게 맞지 않을까요? 

결국 제 생각에 앞으로의 핵심은 지식 베이스로 데이터가 쌓여 추론을 할 수 있는 알고리즘은 만드냐가 핵심인 것 타습니다.

A7 (지성). 맞습니다 ! 논리적인 구성에대한 여부를 보면 문맥을 이해한다 할수 있지만, 사람은 문맥을 이해할때 환경적, 경험적 요소를 같이 고려하여 받아드리기에 위와같은 생각을 했던거 같습니다. 추론의 영역은 인공지능이 아직은 해결하지 못하는 문제입니다 ㅠ 말씀하신대로 어떻게하면 지식 베이스로 데이터를 쌓아 추론할지가 가장 화두가 될거 같습니다 ! :D

Q8 (선민). pretraining 중 사람이 평생 보는 text보다 많은 data를 사용하지만, data의 효율성이 떨어진다는 것을 한계로 제시했는데 사람만큼 효율적으로 학습했다고 판단할 수 있는 절대적인 지표가 있을까요?

A8 (정훈). 현재 딥러닝 논문 중 효율적으로 학습했다라는 지표를 언급한 것은 저는 보지 못했습니다. 데이터를 통해 지식을 쌓는 효율성이 증가하려면 결국은 단어의 개념과 지식을 체계화하는 과정이 필요할 것 같습니다. 결국 체계없이 데이터를 쌓으니 이런문제가 발생하는게 아닐까? 싶은 생각도 드네요.

하지만, 문득 NN 기반 Word Embedding 학습 방식을 생각해보면 각 단어들의 관계성도 벡터로 표현이 되기 때문에 어떻게 보면 지식이 체계화되어 쌓이는 것이 아닐까란 생각도 드네요. 예를 들어, word2vec에 서울, 부산, 대구라는 지역명이 유사도 점수가 높은 것처럼요. 그렇다면 Pre-training 모델에서 부족한 것은 아직도 데이터가 부족하거나 알고리즘의 파리미터 수가 부족한 것은 아닐지 그런 생각이 듭니다.

Q9 (선민). task에 대한 prompt가 주어지면 성능이 향상된다고 하는데 mixture model 개념과 비슷한 부분이 있을까요? 2개 이상의 downstream task를 지원하는 mixture와 GPT의 meta learning의 개념을 더 명확하게 이해하고 싶어 질문드립니다.

A9 (쯔위). 

Q9 (지은). trillion에 가까운 Common Crawl 데이터셋을, 175B 단위의 모델로 돌렸음에도 불구하고 '아직은' overfitting을 경험하지 못했다고 하는데요. 보통 데이터셋의 크기 대비, 모델의 파라미터가 이 정도 수준 (e.g. 데이터셋 크기가 100일때, 모델 파라미터가 1만이면 overfit 가능성이 높다!..같은) 이면 overfit 될 가능성이 있다고 하는 기준에 대해서 논의된 것이 있을까요? 

A9 (쯔위).
