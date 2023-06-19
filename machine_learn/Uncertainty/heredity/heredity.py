import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():
    # Check for proper usage
    # if len(sys.argv) != 2:
    #     sys.exit("Usage: python heredity.py data.csv")
    # people = load_data(sys.argv[1])
    people = load_data("data/family0.csv")

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def gene_number(peo, one_gene, two_genes):
    if peo in one_gene:
        gene_num = 1
    elif peo in two_genes:
        gene_num = 2
    else:
        gene_num = 0
    return gene_num


def trait_not(peo, have_trait):
    if peo in have_trait:
        return True
    else:
        return False


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """

    joint_proba = 1
    for peo in people.keys():
        gene_num = gene_number(peo, one_gene, two_genes)
        dad = people[peo]["father"]
        mom = people[peo]["mother"]

        # the percentage of having one gene or two genes or no gene
        if dad == None and mom == None:
            joint_proba *= PROBS["gene"][gene_num]
        else:
            from_parent = []
            for par in [dad, mom]:
                par_gene_num = gene_number(par,one_gene,two_genes)
                # the percentage of the gene from parent
                if par_gene_num == 0:
                    from_parent.append(PROBS["mutation"])
                elif par_gene_num == 1:
                    from_parent.append(0.5 + 0.5 * PROBS["mutation"])
                else:
                    from_parent.append(1 - PROBS["mutation"])
            if gene_num == 0:
                joint_proba *= (1 - from_parent[0]) * (1 - from_parent[1])
            elif gene_num == 1:
                joint_proba *= from_parent[0] * (1 - from_parent[1]) + (1 - from_parent[0]) * from_parent[1]
            else:
                joint_proba *= from_parent[0] * from_parent[1]

        # the percentage of having the trais or not
        joint_proba *= PROBS["trait"][gene_num][trait_not(peo, have_trait)]

    return joint_proba


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for peo in probabilities.keys():
        gene_num = gene_number(peo, one_gene, two_genes)
        trait = trait_not(peo, have_trait)
        probabilities[peo]["gene"][gene_num] += p
        probabilities[peo]["trait"][trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for peo in probabilities.keys():
        for kind in probabilities[peo].keys():
            all_prob = sum(probabilities[peo][kind].values())
            probabilities[peo][kind].update((key, val / all_prob) for key, val in probabilities[peo][kind].items())



if __name__ == "__main__":
    main()