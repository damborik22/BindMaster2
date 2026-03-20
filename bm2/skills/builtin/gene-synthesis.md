---
name: gene-synthesis
description: Gene synthesis vendors, costs, and construct design
keywords: [gene, synthesis, twist, IDT, construct, codon, clone, order, cost, vendor]
---
# Gene Synthesis Guide

## Vendor Comparison

| Vendor | Cost (~300bp) | Turnaround | Notes |
|--------|-------------|------------|-------|
| Twist Bioscience | ~$100-150 | 2-3 weeks | Best bulk pricing |
| IDT gBlocks | ~$100-200 | 1 week | Fast |
| GenScript | ~$150-300 | 2-4 weeks | Full cloning service |

## Construct Design

### E. coli: His6-TEV-binder in pET vector
NdeI - MHHHHHH - ENLYFQS - [binder] - Stop - XhoI

### Mammalian: signal-binder-TEV-His6
IL2ss - [binder] - ENLYFQS - HHHHHH - Stop in pcDNA3.4

## Tips
- Codon optimize for host (E. coli or HEK293)
- Avoid rare codons, internal restriction sites, homopolymers > 6nt
- 20+ genes at Twist: ~30-40% volume discount
- Gibson assembly: add ~40bp adapters per end
