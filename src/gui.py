from tkinter import *
from tkinter import ttk
import os

class Input:
    '''
    Class containing the GUI.
    '''

    def __init__(self, root):
        '''
        This function creates the widgets for the GUI.
        '''

        root.title("Boolean Cell Cycle Analysis")

        mainframe = ttk.Frame(root, padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
       
        self.model = StringVar()
        model_entry = ttk.Combobox(mainframe, width=15, textvariable=self.model, values=("model01", "model02", "model03"), state='readonly')
        model_entry.set("model01")
        model_entry.grid(column=2, row=1, sticky=(W, E))
        ttk.Label(mainframe, text="Represents the model to use. Available models: model01, model02 and model03.").grid(column=3, row=1, sticky=W)

        self.filter = BooleanVar()
        filter_states = ttk.Checkbutton(mainframe, text='Enable to use filter states', variable=self.filter).grid(column=3, row=2, sticky=W)

        self.custom = BooleanVar()
        custom_states = ttk.Checkbutton(mainframe, text='Enable to use custom states', variable=self.custom).grid(column=3, row=3, sticky=W)

        self.singleit = IntVar()
        single_entry = ttk.Entry(mainframe, width=7, textvariable=self.singleit)
        single_entry.grid(column=2, row=4, sticky=(W, E))
        ttk.Label(mainframe, text="Enter the number of single iterations the program should run.").grid(column=3, row=4, sticky=W)

        self.doubleit = IntVar()
        double_entry = ttk.Entry(mainframe, width=7, textvariable=self.doubleit)
        double_entry.grid(column=2, row=5, sticky=(W, E))
        ttk.Label(mainframe, text="Enter the number of double iterations the program should run.").grid(column=3, row=5, sticky=W)

        ttk.Button(mainframe, text="Run", command=self.run).grid(column=3, row=6, sticky=W)

        for child in mainframe.winfo_children(): 
            child.grid_configure(padx=5, pady=5)

        model_entry.focus()
        
    def run(self, *args):
        '''
        Run async_perturb_test.py with the GUI inputs.
        '''
        if self.filter.get():
            self.filter.set(True)
        if self.custom.get():
            self.custom.set(True)
        root.destroy()
        try:
            print(f"python async_perturb_test.py {self.model.get()} {self.filter.get()} {self.custom.get()} {self.singleit.get()} {self.doubleit.get()}")
            os.system(f"python async_perturb_test.py {self.model.get()} {self.filter.get()} {self.custom.get()} {self.singleit.get()} {self.doubleit.get()}")
        except:
            print("ERROR: Incorrect inputs. Make sure you use integers.")
        exit(0)

root = Tk()
Input(root)
root.mainloop()