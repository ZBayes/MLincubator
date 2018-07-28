# MLincubator

## An Experimental Machine Learning Framework
The name looks domineering, but it is not that confusing technologically. It is constructed just based on some experience in the experiment and modeling process.

**MLincubator is a frameork to manage your idea and code for better machine-learning experiment.**

Basically, since the idea has been basically raised, some of the functions have not yet been fully realized. Suggestion is welcomed.

### Basic Requirements for Experimentation
- multiple programs  
    - Several programs are generated during your experiment, so it is important to store your programs and your result. 
- Multiple parts
    - In the various programs, some parts may be the same so we dont need to run again and the model can be save systematically. 
- Log system
    - Record and measure the results of various programs 

### Basic thoughts
- Block
    + Divide multiple step parts and connect each step by recording the process results
- Log
    + Record experimental results for proper comparison
- Quickly package
    + Directly call the stored model for process calculation

### File Management
-data  
|-src_data  
|-model  
|-log  
-src  
|-data_explore  
|-data_process  
|-model  
|-eval  
|-util  
|-flow  