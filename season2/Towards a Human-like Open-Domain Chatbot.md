<span style="color:red">Q1 (정훈).</span>  SSA는 모델 학습이 모두 끝난 후 평가에 사용되는 최종 Metric일까요? 최종 Metric이면 모델은 학습이 끝난 후고 모델들의 성능을 평가하는 도구이기 때문에 직접적으로 모델에 영향을 미친기는 힘들다고 생각하는데 다른 분들의 생각은 어떠하신가요?

'A1 (민상).' 애초에 SSA는 human evaluation metric이기 때문에 시간이나 비용 등의 cost가 클 수밖에 없을 것 같아요. 이 논문에서 그랬듯 이미 개발이 끝난 모델들을 비교하는 데는 유용하게 쓰일 수 있겠지만, 모델을 개발하는 과정에서 빠르고 간단한 evaluation이 필요하다면 SSA는 쓰이기 힘들 것 같네요.

A2 (선민). 저도 직접적으로 모델에 영향을 미치기에는 어렵다고 생각합니다. 본 논문에서는 SSA를 통해 비교적 객관적인 sensibleness와 비교적 주관적인 specificity를 통해 챗봇의 human-likeness를 평가할 수 있 있는 지표를 소개했다는 점에서 의의가 있다고 생각합니다.

- Q2 (정훈). 디코딩 과정은 N개의 독립적인 후보 답변 중 final output의 가장 가까운 값을 하나 출력한다고 하는데 후보 중 선택하는 기준과 방법은 무엇일까요?

  >- A2 (민상). Table 2에 달린 설명을 보니, $Score=\frac {\log P} {T}$를 계산해서 가장 높은 score를 가진 후보를 선택한 것 같아요!

Q3 (민상). Perplexity에 대해 배울 때면 으레 'perplexity가 낮다고 해서 인간이 느끼기에 좋은 모델은 아닐 수 있다'는 말을 듣고는 하는데요. 이 논문에서 제시한 SSA는 perplexity와의 correlation이 높다는 게 참 흥미로웠습니다. Sensibleness와 specificity가 perplexity와 높은 correlation을 가진 이유를 설명할 수 있을까요?

A3 (정훈). 저도 이 논문의 가장 핵심은 SSA와 Perplexity과 높은 correlation을 가진다고 주장하는 부분이라고 생각합니다. 우선 기존의 Generation 방식의 알고리즘 대부분 loss 계산을 perplexity로 했었습니다. 사실 이 metric을 과거에 선택것은 다른 방법이 없기 때문이라고 생각합니다. 물론 사람이 직접 수작업으로 보고 평가하는게 제일 좋지만 현실적으로 한계가 있겠죠?

그래서 Perplextiy 같은 automatic metric을 사용했었는데 이 방법이 실제 좋은 챗봇 모델을 만드는 loss function으로 적합하다는 사실을 뒷받침하는 것이라 생각합니다. 흔히 generation 모델의 가장 큰 문제점은 metric이라고 하는데 이 부분을 perplextiy를 써도 괜찮다는 것을 증명해주었다고 합니다.

사실 저도 왜 corrleation이 높은지 이유를 명확히 설명하지는 않았다고 생각합니다. 제 개인적인 생각으로 구글에서 해당 모델을 학습할 때 사용한 데이터의 퀄리티가 좋기 때문에 더 밀접한 관계를 보인 것이라 생각합니다. 왜냐하면 Perplexity 수식을 보면 각 단어의 발생 확률을 체인룰로 계산해서 점수를 도출하기 때문에 결국은 학습이나 테스트 데이터에 맞춰져서 모델이 생성될 것 입니다. 그렇다면 데이터가 좋기 때문에 모델도 좋겠죠? 그런 상태에서 최종 테스트 데이터로 평가했을 때 SSA가 좋지 않을까란 생각을 합니다. 

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/162a36ee-3221-40e5-83b0-3f272b687f6b/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/162a36ee-3221-40e5-83b0-3f272b687f6b/Untitled.png)

Q4 (민상). Input으로 들어간 메시지가 어떤 형태인지 잘 감이 안 오네요ㅠ 메시지에 대한 대답을 자식 노드로 삼는 방식으로 트리 구조를 만들었다는데, 대화에서 하나의 메시지에 대해 여러가지 대답이 존재했다는 걸까요? 만약 그렇다면 특정 대답이 어떤 메시지에 대한 것인지 어떻게 구별했을까요? (처음에는 챗봇처럼 시간에 따라 선형적으로 문답을 주고받는 형식이리라 생각했는데, 그런 형태면 굳이 트리를 사용해야 했나 싶기도 하고...)

A4 (선민). train 할 때 (context, response) pair 형태인데 root에서 7번 context를 추출한다는 것 같습니다. 7을 선택한 이유는 충분한 context 양과 메모리 한계 를 고려했을 때 가장 적절한 수라고 합니다.

A4(소연) 저도 이부분 궁금해요! 제가 이해한 선으로는 섹션 3.1에서 설명하듯, social media conversation과 같은 open dataset에서 , 시작점이 될 가능성있는 (임의의) 부분을 root로 두고 해당 root에 따라 쭉 발생하는 path 하나하나가 RL에서의 a single training trajectory와 같은? 싱글 데이터 뭉치..? 하나하나가 되는 것이라 이해했습니다..

Q5 (민상). 이번 논문을 읽으며 도움이 된 참고자료가 있다면 모아봐도 좋을 것 같아요!

A5 (민상). 저는 이번 논문을 읽으며 완전 기초적인 개념들부터 복습을 해야 했네요ㅎㅎㅎ

End-to-end model에 대한 기본 개념: [https://towardsdatascience.com/e2e-the-every-purpose-ml-method-5d4f20dafee4](https://towardsdatascience.com/e2e-the-every-purpose-ml-method-5d4f20dafee4)

Transformer 이해하기 (인코더와 디코더가 무엇인지?): [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

자연어 생성 evaluation에 자주 쓰이는 metrics: [https://medium.com/explorations-in-language-and-learning/metrics-for-nlg-evaluation-c89b6a781054](https://medium.com/explorations-in-language-and-learning/metrics-for-nlg-evaluation-c89b6a781054)

Evolved Transformer 아키텍처: [https://arxiv.org/abs/1901.11117](https://arxiv.org/abs/1901.11117)

A5 (지은). 스캐터랩에서 나온 자료로, 가장 도움이 되었습니다. [https://speakerdeck.com/scatterlab/towards-a-human-like-open-domain-chatbot-review?slide](https://speakerdeck.com/scatterlab/towards-a-human-like-open-domain-chatbot-review?slide)

A5. (소연) 감사합니다..

Q6 (선민) response 후보 중 이전에 response한 것이 있으면 제외하여 SSA를 올렸다고 하는데 이것은 fair한 평가일까요? 하나의 알고리즘으로 보고 fair한 평가가 되는 것인가요? 

Q7 (선민) backgound로 올려주신 자료에서 Retrieval Based는 오픈 도메인에 특화된 데이터를 구성한다는 것이 다양한 도메인에 대한 데이터를 구성한다는 것인가요? 이것이 모호한 대답을 만드는데 영향을 미칠 수 있을까요? 

A7 (정훈). Open-domain에서 Retrieval Based 방식으로 구현할 때 사용하는 데이터는 말씀하신데로 다양한 도메인 데이터로 구성한다는 뜻입니다. 예를 들어, 가장 대표적으로 최근의 "이루다"라는 챗봇 또한 이 방식으로 구현되었습니다. "이루다"에서 수집한 데이터는 핑퐁의 자사 데이터와 외부 데이터를 수집하는 식으로 구성되었습니다. 

예를 들어 일상생활과 관련된 대화. "너 밥 먹었어" - "나 오늘 마라탕 먹었어", "시험 잘 봤니?" - "망쳤어 ㅠㅠ"와 같은 도메인이 정해져 있지 않은 대화들이 무수히 많이 구성되어 있습니다. 그렇기 때문에 얼핏 보면 대부분의 대화 주제에 응대하는 것으로 보이구요.

이것이 모호한 대답을 만드는데 영향을 미칠 수 있습니다. 예를 들어, 새로운 질문이 들어왔을 때 기존 질문과 비교했을 때 유사도 점수가 전부다 낮은 경우에 "나는 잘 모르겠는데"라는 모호한 대답을 출력하게 하는 경우가 대부분입니다. 이런 경우는 모호한 대답을 만드는데 영향을 미친다고 할 수 있겠죠?

물론 해당 문제는 Generation 방식에서도 나타날 수 있습니다. 학습데이터에 "나는 잘 모르겠는데"라는 답변이 많이 포함되면 그 대답을 자주 하도록 챗봇은 구성 될 것 입니다.

 

Q8 (지성).  human evaluation metric 이다보니 train dataset quality 또는, 내용적, 문화적인 환경에 등의 bias에 따라 논문에서 말하는 Sensibleness 조건 (논리적, 일관성 여부, 특히 상식적인지 등)에 부합하는 기준(의미(?)가 다르게 적용되어 영향을 줄것 같다는 생각도 드네요. 다른분들은 어떻게 생각하실지 궁금합니다 (개인적으로는 철학(?)적인 물음이 많이 생기는 paper인 것 같네요)  

A8 (지은).  충분히 있을 수 있다고 봅니다. 예를 들어서 인도에서는 계급 차별이 당연시 되어서, 브라만 계급을 대표하는 이름이 있을 경우 갑자기 존댓말을 한다던가 어법 자체가 완전히 바뀔 수도 있을 것 같은데요. 이런식으로 트레이닝된 챗봇이 미국같은 곳에서도 보편적으로 받아들여질 것 같진 않습니다. bias가 너무 크고 처음부터 다 다시 트레이닝해야 하지 않을까 싶기에, general한 측면이 굉장히 강조될 것 같은데.. 문제는 이 general의 기준을 developed countries에 맞출 경우, 아직 개발도상국이거나 가난한 국가들에서 서비스를 하려고 할 때, 그 국가들 시점에서 또 bias되어 있다고 생각할 수 있을 것 같습니다. 

A8 (소연). 저도 해당 논문 읽으면서, 그리고 nlp task 에서 이러한 대화를 이루는 데이터셋은 어느 문화권? 기준의 상식?을 적용하여 데이터셋을 구축하는건지 궁금했습니다. 이미지보다도 "언어"가 문화권에 영향을 많이 주기 때문에 generalization, fairness 이슈가 더 많이 발생하는 것 같고, 서비스 제공 시에도 더 많은 localization이 필요할 것같은데 그런 생태계가 궁금하네요!

Q8 (지은).  이런 챗봇들은 사람이 질문하는 것에 대해서 '모두 알고 있다'는 가정하에 진행되는데요. 실제 현실의 사람은 '얼불노'라고 하면 '그게 뭔데?'하고 물어볼 수도 있는데, Meena에서는 도메인 지식이 전부 갖춰져 있다는 전제하에 대화가 진행이 됩니다. 이런식의 approach가 혹시 '덜 인간다운 점'일 수 있다고 지적한 논문같은게 있을까요?

Q9 (소연).  제가 AI 시스구축에 대한 생태계와 nlp 모델의 메모리/컴퓨팅 연산양에 대한 감이 좀 부족해서 궁금한 질문입니다. vision보다는 NLP task가 여러 서비스에 더 접목될 가능성이 높다고 보는데, **section 2.5**에서의 762M 정도면, 1) 일반적인 "서비스 제공 입장"에서 적정한 수준의 모델 크기인지, 2) 서비스를 제하고 보통 nlp 에서 SOTA 또는 주류를 이루는 모델 크기 기준 762M가 어느 정도  준수한 크기(메모리)수준에 속하는 것인지 궁금합니다..(e.g, vision task에서 SOTA는 다양하게 있겠지만 resnet, vgg를 백본으로 쓰는 경우가 많고, 그런 모델에 대해 config에 따라 대충 어느정도 사이즈다를 감안하듯)

A9 (정훈). 1)우선 서비스 제공 입장에서 적정한 수준의 모델 크기는 inference 했을 때 지연 속도가 실서비스에 영향을 주지 않을 정도 일 것 같습니다. 저희 톡집사 같은 경우에는 질문이 입력했을 때 모델 연산 속도가 대략 0.05sec 이하로 맞추는 것으로 지침을 잡고 있습니다. 이 부분은 모델 크기가 커짐에 따라 연산 속도가 오래 걸릴 수 있겠죠?

2) 준수한 크기의 메모리라.. 연구단에서는 사실 모델의 크기가 크든 작든 중요하지는 않은 것 같습니다. 점점 대용량 모델이 발전하고 있고 사실 연구에서는 latency를 신경쓰지 않기 때문이죠. 물론 경량화 연구나 실제 서비스 되는 상황을 가정하는 논문에서는 예외겠죠? 

우선 현재 주류 모델이 Transformer 기반의 BERT 같은 모델이 거의 기본으로 깔려있기 때문에 대부분의 모델크기가 700MB 이상은 되는 것 같습니다.

Q10 (소연).  논문에서도, 그리고 위의 지은님이 올려주신 스캐터 자료에서도, beam search와 sample and rank의 성능에 대한 비교가 나옵니다. 저는 왜 sample and rank가 더 좋은 결과가 나와보이는건지 대한 직관적 해석, 의견이 궁금합니다. (diverse하고 높은 퀄리티의 답변이 "sample" 이라는 randomness가 적용돼서인지,..? ) 

더불어 sample and rank의 N=20, T=0.88이라는 hyperparameter 값은 여러 Hyperparmeter로 서치하면서 얻게 되는 값인가요? 특히나 0.88 같은 값은.. 많이 사용되는 configuration 값이라기보다는  model/ task specific 한 값으로 gpu 노가다로 얻어지는 값처럼 보여서요.
