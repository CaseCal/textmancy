import dotenv
from textmancy.extractor import Extractor
from textmancy.consolidator import Consolidator
from textmancy.annotator import Annotator
from textmancy.targets import Character

dotenv.load_dotenv()

# Get text and split into pages
with open("sample_data/christmas_carol.txt", encoding="utf-8") as f:
    text = f.read()
    # pages = text.split("\n\n")
    paragraphs = text.split("\n")
    pages = [paragraphs[i : i + 10] for i in range(0, len(paragraphs), 10)]

# Extract and consolidate characters

extractor = Extractor(
    target_class=Character,
    target_num=10,
    target_examples=[],
    model="gpt-4-1106-preview",
)

consolidator = Consolidator(
    target_class=Character,
    target_num=10,
    model="gpt-4-1106-preview",
)

results = extractor.extract(text, chunk_size=12000, number=3)
print(f"{len(results)} results found")
results = consolidator.consolidate(results)
print(f"{len(results)} results consolidated")

for r in results:
    print(r)
    print()


# Annotate pages with characters
annotater = Annotator(
    targets=results,
    model="gpt-4-1106-preview",
)
annotations = [annotater.annotate(page) for page in pages[:5]]

# Show annotations
for page, annotation in zip(pages, annotations):
    print(page)
    print(annotation)
    print([results[i].name for i in annotation])
    print()
