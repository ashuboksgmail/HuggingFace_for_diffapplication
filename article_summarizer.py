#import pipeline
from transformers import pipeline

#initializing modal for Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = "Artificial intelligence (AI) applications in education are on the rise and have received a lot" \
          " of attention in the last couple of years. AI and adaptive learning technologies are prominently featured as " \
          "important developments in educational technology in the 2018 Horizon report (Educause, 2018), with a time to adoption " \
          "of 2 or 3 years. According to the report, experts anticipate AI in education to grow by 43% in the period 2018–2022, " \
          "although the Horizon Report 2019 Higher Education Edition (Educause, 2019) predicts that AI applications related to" \
          " teaching and learning are projected to grow even more significantly than this. Contact North, a major Canadian non-profit " \
          "online learning society, concludes that “there is little doubt that the [AI] technology is inexorably linked to the future " \
          "of higher education” (Contact North, 2018, p. 5). With heavy investments by private companies such as Google, which acquired " \
          "European AI start-up Deep Mind for $400 million, and also non-profit public-private partnerships such as the German Research " \
          "Centre for Artificial IntelligenceFootnote1 (DFKI), it is very likely that this wave of interest will soon have a significant " \
          "impact on higher education institutions (Popenici & Kerr, 2017). The Technical University of Eindhoven in the Netherlands, " \
          "for example, recently announced that they will launch an Artificial Intelligence Systems Institute with 50 new professorships" \
          " for education and research in AI"

summery = summarizer(article, max_length=100, min_length=80)

print(summery)

