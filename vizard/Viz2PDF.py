from matplotlib.backends.backend_pdf import PdfPages

class Viz2PDF():
    def __init__(self, pdf_filepath):
        self.pdf_filepath = pdf_filepath
    
    def __call__(self, plots):
        with PdfPages(self.pdf_filepath) as pp:
            for plot in plots:
                if isinstance(plot, list):
                    for pl in plot:
                        pp.savefig(pl)
                else:
                    pp.savefig(plot)
    