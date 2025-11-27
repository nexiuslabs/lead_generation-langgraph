#Sample code

import requests
import json

url = "https://deepsearch.jina.ai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer jina_6ca65401d6024aef847af85f8394d2ecSIXuMMh41XupIlb5Q27DjJ2gIfnb"
}
data = {
    "model": "jina-deepsearch-v1",
    "messages": [
        {
            "role": "user",
            "content": "Hi!"
        },
        {
            "role": "assistant",
            "content": "Hi, how can I help you?"
        },
        {
            "role": "user",
            "content": "what's the latest blog post from jina ai?"
        }
    ],
    "stream": True,
    "reasoning_effort": "low",
    "max_returned_urls": "50",
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "type": "object",
            "properties": {
                "numerical_answer_only": {
                    "type": "number"
                }
            }
        }
    },
    "bad_hostnames": [
        "*.gov.sg",
        "*.gov",
        "*.xyz"
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.json())



## Response
{
  "id": "1742181758589",
  "object": "chat.completion.chunk",
  "created": 1742181758,
  "model": "jina-deepsearch-v1",
  "system_fingerprint": "fp_1742181758589",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "The latest blog post from Jina AI is titled \"Snippet Selection and URL Ranking in DeepSearch/DeepResearch,\" published on March 12, 2025 [^1]. This post discusses how to improve the quality of DeepSearch by using late-chunking embeddings for snippet selection and rerankers to prioritize URLs before crawling. You can read the full post here: https://jina.ai/news/snippet-selection-and-url-ranking-in-deepsearch-deepresearch\n\n[^1]: Since our DeepSearch release on February 2nd 2025 we ve discovered two implementation details that greatly improved quality In both cases multilingual embeddings and rerankers are used in an in context manner operating at a much smaller scale than the traditional pre computed indices these models typically require  [jina.ai](https://jina.ai/news/snippet-selection-and-url-ranking-in-deepsearch-deepresearch)",
        "type": "text",
        "annotations": [
          {
            "type": "url_citation",
            "url_citation": {
              "title": "Snippet Selection and URL Ranking in DeepSearch/DeepResearch",
              "exactQuote": "Since our DeepSearch release on February 2nd 2025, we've discovered two implementation details that greatly improved quality. In both cases, multilingual embeddings and rerankers are used in an _\"in-context\"_ manner - operating at a much smaller scale than the traditional pre-computed indices these models typically require.",
              "url": "https://jina.ai/news/snippet-selection-and-url-ranking-in-deepsearch-deepresearch",
              "dateTime": "2025-03-13 06:48:01"
            }
          }
        ]
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 169670,
    "completion_tokens": 27285,
    "total_tokens": 196526
  },
  "visitedURLs": [
    "https://github.com/jina-ai/node-DeepResearch/blob/main/src/utils/url-tools.ts",
    "https://huggingface.co/jinaai/jina-embeddings-v3",
    "https://github.com/jina-ai/reader",
    "https://zilliz.com/blog/training-text-embeddings-with-jina-ai",
    "https://threads.net/@unwind_ai/post/DGmhWCVswbe/media",
    "https://twitter.com/JinaAI_/status/1899840196507820173",
    "https://jina.ai/news?tag=tech-blog",
    "https://docs.llamaindex.ai/en/stable/examples/embeddings/jinaai_embeddings",
    "https://x.com/jinaai_",
    "https://x.com/JinaAI_/status/1899840202358784170",
    "https://tracxn.com/d/companies/jina-ai/__IQ81fOnU0FsDpagFjG-LrG0DMWHELqI6znTumZBQF-A/funding-and-investors",
    "https://jina.ai/models",
    "https://linkedin.com/posts/imohitmayank_jinaai-has-unveiled-the-ultimate-developer-activity-7300401711242711040-VD64",
    "https://medium.com/@tossy21/trying-out-jina-ais-node-deepresearch-c5b55d630ea6",
    "https://huggingface.co/jinaai/jina-clip-v2",
    "https://arxiv.org/abs/2409.10173",
    "https://milvus.io/docs/embed-with-jina.md",
    "https://seedtable.com/best-startups-in-china",
    "https://threads.net/@sung.kim.mw/post/DGhG-J_vREu/jina-ais-a-practical-guide-to-implementing-deepsearchdeepresearchthey-cover-desi",
    "https://elastic.co/search-labs/blog/jina-ai-embeddings-rerank-model-open-inference-api",
    "http://status.jina.ai/",
    "https://apidog.com/blog/recreate-openai-deep-research",
    "https://youtube.com/watch?v=QxHE4af5BQE",
    "https://sdxcentral.com/articles/news/cisco-engages-businesses-on-ai-strategies-at-greater-bay-area-2025/2025/02",
    "https://aws.amazon.com/blogs/machine-learning/build-rag-applications-using-jina-embeddings-v2-on-amazon-sagemaker-jumpstart",
    "https://reddit.com/r/perplexity_ai/comments/1ejbdqa/fastest_open_source_ai_search_engine",
    "https://search.jina.ai/",
    "https://sebastian-petrus.medium.com/build-openais-deep-research-open-source-alternative-4f21aed6d9f0",
    "https://medium.com/@elmo92/jina-reader-transforming-web-content-to-feed-llms-d238e827cc27",
    "https://openai.com/index/introducing-deep-research",
    "https://python.langchain.com/docs/integrations/tools/jina_search",
    "https://varindia.com/news/meta-is-in-talks-for-usd200-billion-ai-data-center-project",
    "https://varindia.com/news/Mira-Murati%E2%80%99s-new-AI-venture-eyes-$9-billion-valuation",
    "https://53ai.com/news/RAG/2025031401342.html",
    "https://arxiv.org/abs/2409.04701",
    "https://bigdatawire.com/this-just-in/together-ai-raises-305m-series-b-to-power-ai-model-training-and-inference",
    "https://github.blog/",
    "https://cdn-uploads.huggingface.co/production/uploads/660c3c5c8eec126bfc7aa326/MvwT9enRT7gOESHA_tpRj.jpeg",
    "https://cdn-uploads.huggingface.co/production/uploads/660c3c5c8eec126bfc7aa326/JNs_DrpFbr6ok_pSRUK4j.jpeg",
    "https://app.dealroom.co/lists/33530",
    "https://api-docs.deepseek.com/news/news250120",
    "https://sdxcentral.com/articles/news/ninjaone-raises-500-million-valued-at-5-billion/2025/02",
    "https://linkedin.com/sharing/share-offsite?url=https%3A%2F%2Fjina.ai%2Fnews%2Fa-practical-guide-to-implementing-deepsearch-deepresearch%2F",
    "https://twitter.com/intent/tweet?url=https%3A%2F%2Fjina.ai%2Fnews%2Fa-practical-guide-to-implementing-deepsearch-deepresearch%2F",
    "https://platform.openai.com/docs/api-reference/chat/create",
    "https://mp.weixin.qq.com/s/-pPhHDi2nz8hp5R3Lm_mww",
    "https://huggingface.us17.list-manage.com/subscribe?id=9ed45a3ef6&u=7f57e683fa28b51bfc493d048",
    "https://automatio.ai/",
    "https://sdk.vercel.ai/docs/introduction",
    "https://app.eu.vanta.com/jinaai/trust/vz7f4mohp0847aho84lmva",
    "https://apply.workable.com/huggingface/j/AF1D4E3FEB",
    "https://facebook.com/sharer/sharer.php?u=https%3A%2F%2Fjina.ai%2Fnews%2Fa-practical-guide-to-implementing-deepsearch-deepresearch%2F",
    "https://facebook.com/sharer/sharer.php?u=http%3A%2F%2F127.0.0.1%3A3000%2Fen-US%2Fnews%2Fsnippet-selection-and-url-ranking-in-deepsearch-deepresearch%2F",
    "https://reddit.com/submit?url=https%3A%2F%2Fjina.ai%2Fnews%2Fa-practical-guide-to-implementing-deepsearch-deepresearch%2F",
    "https://apply.workable.com/huggingface",
    "https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fjina.ai%2Fnews%2Fa-practical-guide-to-implementing-deepsearch-deepresearch%2F",
    "https://news.ycombinator.com/submitlink?u=http%3A%2F%2F127.0.0.1%3A3000%2Fen-US%2Fnews%2Fsnippet-selection-and-url-ranking-in-deepsearch-deepresearch%2F",
    "https://docs.github.com/site-policy/privacy-policies/github-privacy-statement",
    "https://discord.jina.ai/",
    "https://docs.github.com/site-policy/github-terms/github-terms-of-service",
    "https://bigdatawire.com/this-just-in/qumulo-announces-30-million-funding",
    "https://x.ai/blog/grok-3",
    "https://m-ric-open-deep-research.hf.space/",
    "https://youtu.be/sal78ACtGTc?feature=shared&t=52",
    "https://mp.weixin.qq.com/s/apnorBj4TZs3-Mo23xUReQ",
    "https://perplexity.ai/hub/blog/introducing-perplexity-deep-research",
    "https://githubstatus.com/",
    "https://github.blog/changelog/2021-09-30-footnotes-now-supported-in-markdown-fields",
    "https://openai.com/index/introducing-operator",
    "mailto:support@jina.ai",
    "https://resources.github.com/learn/pathways",
    "https://status.jina.ai/",
    "https://reuters.com/technology/artificial-intelligence/tencents-messaging-app-weixin-launches-beta-testing-with-deepseek-2025-02-16",
    "https://scmp.com/tech/big-tech/article/3298981/baidu-adopts-deepseek-ai-models-chasing-tencent-race-embrace-hot-start",
    "https://microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks",
    "javascript:UC_UI.showSecondLayer();",
    "https://resources.github.com/",
    "https://storm-project.stanford.edu/research/storm",
    "https://blog.google/products/gemini/google-gemini-deep-research",
    "https://youtu.be/vrpraFiPUyA",
    "https://chat.baidu.com/search?extParamsJson=%7B%22enter_type%22%3A%22ai_explore_home%22%7D&isShowHello=1&pd=csaitab&setype=csaitab&usedModel=%7B%22modelName%22%3A%22DeepSeek-R1%22%7D",
    "https://app.dover.com/jobs/jinaai",
    "http://localhost:3000/",
    "https://docs.cherry-ai.com/",
    "https://en.wikipedia.org/wiki/Delayed_gratification",
    "https://support.github.com/?tags=dotcom-footer",
    "https://docs.jina.ai/",
    "https://skills.github.com/",
    "https://partner.github.com/",
    "https://help.x.com/resources/accessibility",
    "https://business.twitter.com/en/help/troubleshooting/how-twitter-ads-work.html",
    "https://business.x.com/en/help/troubleshooting/how-twitter-ads-work.html",
    "https://support.twitter.com/articles/20170514",
    "https://support.x.com/articles/20170514",
    "https://t.co/jnxcxPzndy",
    "https://t.co/6EtEMa9P05",
    "https://help.x.com/using-x/x-supported-browsers",
    "https://legal.twitter.com/imprint.html"
  ],
  "readURLs": [
    "https://jina.ai/news/a-practical-guide-to-implementing-deepsearch-deepresearch",
    "https://github.com/jina-ai/node-DeepResearch",
    "https://huggingface.co/blog/open-deep-research",
    "https://jina.ai/news/snippet-selection-and-url-ranking-in-deepsearch-deepresearch",
    "https://x.com/jinaai_?lang=en",
    "https://jina.ai/news",
    "https://x.com/joedevon/status/1896984525210837081",
    "https://github.com/jina-ai/node-DeepResearch/blob/main/src/tools/jina-latechunk.ts"
  ],
  "numURLs": 98
}
