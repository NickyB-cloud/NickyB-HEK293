NickyB-HEK293: HEK293 TLR4 REPORTER ASSAY PIPELINES
===================================================
by Nicholas B. Franks, PhD

Python scripts that turn HEK293 TLR4/MD-2 reporter-cell plate reads
(NF-kB -> secreted reporter, read by absorbance) into fold-change figures
and statistics, for purified lipid A fractions (bacterial strain x
thin-layer chromatography Rf band) in human and mouse reporter lines.

The scripts can be repurposed for any HEK293 TLR dose-response assay.

These scripts were used to generate the HEK293 figures in Chapter 4 of my
PhD dissertation. You can find the dissertation through the Tufts
University Tisch Library by searching:

    Nicholas Franks, Impact of Microbial Modification of Lipid A on
    Dendritic Cell Functional Responses

  HEK293_TLR_dose/      most similar to Figures 4.2.1 and 4.2.2
  HEK293_TLR_compare/   most similar to Figure 4.2.3

I also recognize that those scripts are fragile and specific to my own
experiments, so I troubleshot with Claude Code to make a hardened, more
general pipeline (HEK293_TLR_Flex/). I stress tested it on made-up data,
which comes alongside it in an Example_Data folder. If you follow
workflow.txt you should be able to recreate the example.pdf that comes
along with it.


Folders
-------
HEK293_TLR_dose/      Dose-response figures (human vs mouse) from a tidy
                      fold-change table made by hand.

HEK293_TLR_compare/   One lipid A preparation (unfractionated "mix") vs its
                      TLC-purified Rf fractions, over timepoints; includes a
                      parser for the plate-reader Excel export.

HEK293_TLR_Flex/      General version for any HEK293 reporter assay
                      (absorbance or fluorescence): set up your plate map
                      once, paste the reader output into Excel, and get a
                      tidy table plus graphs and stats tables.
                      Start here if you are using this with your own data.

Each folder contains

  requirements.txt   Python packages (versions the scripts were run with)

  workflow.txt       what each script does, the order to run them in, the
                     expected input format, and what to change for your data

  *.py               the scripts

HEK293_TLR_Flex/ also has Example_Data/ (four made-up plates and the
example.pdf made from them). No real data files are included.


Getting started
---------------
Python 3.14 was used. For any folder:

    cd HEK293_TLR_Flex             (for example)
    python3 -m venv venv
    venv/bin/pip install -r requirements.txt

Then follow that folder's workflow.txt. To try the Flex pipeline on the
example plates:

    venv/bin/python Parse_Flex_NB.py Example_Data
    venv/bin/python HEK293_Flex_NB.py Example_Data/combined_tidy.csv


License
-------
MIT (see LICENSE). Free to use, modify and share; keep the copyright notice.
