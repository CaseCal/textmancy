import logging

import dotenv

from textmancy.components import LLMAnnotator, PageSegmentor, Processor
from textmancy.targets import Character

dotenv.load_dotenv()

# Print logs to stream
logging.getLogger("textmancy").setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")


# Get text and split into pages
segmentor = PageSegmentor(paragraphs_per_page=10)
with open("sample_data/snows_of_kiliminjaro.txt", encoding="utf-8") as f:
    text = f.read()
    pages = segmentor.segment(text)
    # pages = pages[:5]  # Only process first 5 pages for speed

# Extract and consolidate characters
processor = Processor(target_class=Character)
results = processor.process(pages)

for r in results:
    print(r)
    print()


# Annotate pages with characters
annotater = LLMAnnotator(
    targets=results,
    model="openai:gpt-4o",
)
annotations = [annotater.annotate(page) for page in pages[:5]]

# Show annotations
for page, annotation in zip(pages, annotations):
    print(page)
    print(annotation)
    print([results[i].name for i in annotation])
    print()


# Annotate with TypeSafe's Jev classifier instead (pip install "textmancy[jev]", TYPESAFE_API_KEY)
# from textmancy.components import JevAnnotator
#
# jev = JevAnnotator(targets=results, threshold=0.5, aggregate="max")
# for page in pages[:5]:
#     scores = jev.annotate_scores(page)
#     print({results[i].name: round(p, 2) for i, p in scores.items() if p >= 0.2})
