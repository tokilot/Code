import copy
import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    # if len(sys.argv) != 2:
    #     sys.exit("Usage: python pagerank.py corpus")
    # corpus = crawl(sys.argv[1])
    dire = "corpus0"
    corpus = crawl(dire)
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = {}
    all_pages = list(corpus.keys())
    ran_possibility = (1 - damping_factor) / len(all_pages)
    link_pages = corpus[page]
    link_possibility = damping_factor / len(link_pages)
    for page in all_pages:
        if page in link_pages:
            distribution[page] = ran_possibility + link_possibility
        else:
            distribution[page] = ran_possibility
    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    sample = None
    samples = {key: 0 for key in corpus}
    for i in range(n):
        if sample:
            distri = transition_model(corpus, sample, damping_factor)
            pages = list(distri.keys())
            weights = [distri[page] for page in pages]
            sample = random.choices(pages, weights)[0]
        else:
            sample = random.choice(list(corpus.keys()))

        samples[sample] += 1
    samples.update((key, val / n) for key, val in samples.items())
    return samples


def iterate_pagerank(corpus, damping_factor, confidence=0.001):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    all_pages = list(corpus.keys())
    pagerank = {key: 1 / len(all_pages) for key in all_pages}
    while True:
        pagerank_lst = copy.deepcopy(pagerank)
        for page in all_pages:
            second_con = 0
            for link_page in all_pages:
                if page in corpus[link_page]:
                    second_con += (pagerank_lst[link_page] / len(corpus[link_page]))
                if len(corpus[link_page]) == 0:
                    second_con += (pagerank_lst[link_page]) / len(corpus)
            pagerank[page] = (1 - damping_factor) / len(all_pages) + damping_factor * second_con
        change = max([abs(pagerank[x] - pagerank_lst[x]) for x in pagerank_lst])
        if change < 0.001:
            return pagerank_lst


if __name__ == "__main__":
    main()
