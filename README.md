### Installing pygame and the learning environment

`pip install pygame` 

`git clone https://github.com/ntasfi/PyGame-Learning-Environment`

`cd PyGame-Learning-Environment`

`sudo pip install -e .`

### Training the agents

`python3 train_<agent>.py`


### Testing the agents

For testing we need to change the number of bins appropriately according to the details.txt.

`python3 test_<agent>.py --param_file=<path_to_finalQvalue_file>`

      <Agent>                 <path_to_finalQvalue_file>
      
        mc                       mc_results/1/final_Q.log

       sarsa                    sarsa_results/2/final_Q.log
       
      td_lambda                 td_forward_results/1/final_Q.log

       td_zero                   td_zero_results/final_Q.log
