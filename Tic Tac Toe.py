#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from Ipython.display import clear_output

def display_board(board):
   
   #clear_output()
   
   print('   |   |')
   print(' ' + board[7] + ' | ' + board[8] + ' | ' + board[9])
   print('   |   |')
   print('-----------')
   print('   |   |')
   print(' ' + board[4] + ' | ' + board[5] + ' | ' + board[6])
   print('   |   |')
   print('-----------')
   print('   |   |')
   print(' ' + board[1] + ' | ' + board[2] + ' | ' + board[3])
   print('   |   |')


# In[2]:


test_board = ['#','X','O','X','O','X','O','X','O','X']
display_board(test_board)
#display_board(test_board)


# In[3]:


def player_input():
    
    '''
    Output = (Player 1 marker, Player 2 marker)
    
    '''
    
    marker = ''
    
    while marker != 'X' and marker != 'O':
        marker = input ('Player1: Choose X or O: ').upper()
        
        
    if marker == 'X':
        
        return ('X','O')
    else:
        return ('O','X')


# In[4]:


player1_marker , player2_marker = player_input()


# In[5]:


#Function that takes in a boardlist object, a marker ('X' or 'O') and a desired position (number 1-9) and assigns it to the 
#board


# In[6]:


def place_marker(board,marker,position):
    
    board[position] = marker


# In[7]:


#Test step 3: run the place marker function using test parameters and display the modified board


# In[8]:


place_marker(test_board,'$',8)
display_board(test_board)


# In[9]:


#Function that takes in a board and a mark(X or O) and then checks to see if that mark has won


# In[10]:


def win_check(board, mark):
    
    #Win Tic Tac Toe?
    
    #All Rows, and check to see if they all share the same marker?
    #All columns, check to see if marker matches
    #2 diagonals, check to see if marker matches there
    return ((board[1] == mark and board[2] == mark and board[3] == mark) or #across the bottom
    (board[4] == mark and board[5] == mark and board[6] == mark) or #across the middle
    (board[7] == mark and board[8] == mark and board[9] == mark) or #across the top
    (board[7] == mark and board[4] == mark and board[1] == mark) or # down the middle
    (board[8] == mark and board[5] == mark and board[2] == mark) or # down the middle
    (board[9] == mark and board[6] == mark and board[3] == mark) or # down the right side
    (board[7] == mark and board[5] == mark and board[3] == mark) or # diagonal
    (board[9] == mark and board[5] == mark and board[1] == mark))  #diagonal
    
    


# In[11]:


win_check(test_board,'X')


# In[12]:


#Function that uses the random module to randomly decide which players go first. Random.randint()


# In[13]:


import random

def choose_first():
    
    flip = random.randint(0,1)
    
    if flip == 0:
        return 'Player 1'
    else:
        return 'Player 2'


# In[14]:


#Function that returns a boolean to indicate if there is free space on the board


# In[15]:


def space_check(board,position):
    
    return board[position] == ' '
#return itself will only return a boolean value


# In[16]:


#Function that checks if the board is full. True if full, False otherwise


# In[17]:


def full_board_check(board):
    
    for i in range(1,10):
        if space_check(board, i):
            return False
        
        
    #Board is full if we return True    
    return True    


# In[18]:


#Function that asks the player's next position (as a number 1-9) and then uses the function from step 6 
#to check if its a free position. If it is, then return the position for later use.


# In[19]:


def player_choice(board):
    
    position = 0
    
    while position not in [1,2,3,4,5,6,7,8,9] or not space_check(board,position):
        position = int(input('Choose a position: (1-9) '))
        
    return position    


# In[20]:


#Function to ask the player if they want to play again and return True if they do


# In[21]:


def replay():
    
    choice = input("Play again? Enter Yes or No")
    
    return choice == 'Yes'


# In[22]:


#RUN THE GAME!!!!


# In[ ]:


#While Loop to keep running the game
print('WELCOME TO TIC TAC TOE')


while True:
    
    #Play the Game
    
    ## SET EVERYTHING UP (BOARD, WHOS FIRST, MARKERS )
    
    the_board = [' ']*10
    player1_marker, player2_marker = player_input()
    
    turn = choose_first()
    print(turn + ' will go first')
    
    play_game = input('Ready to play game? y or n? ')
    
    if play_game == 'y':
        game_on = True
    else:
        game_on = False
    
    
    
    
    ## GAME PLAY
    
    while game_on:
        
        if turn == 'Player 1':
            
            # Show the board
            display_board(the_board)
            #Choose a position
            position = player_choice(the_board)
            #Place the marker on the position
            place_marker(the_board,player1_marker,position)
            
            
            #Check if they won
            
            if win_check(the_board,player1_marker):
                display_board(the_board)
                print('PLAYER 1 HAS WON!!')
                game_on = False
            else:
                if full_board_check(the_board):
                    display_board(the_board)
                    print("TIE GAME!!")
                    game_on = False
                else:
                    turn = 'Player 2'
                    
                    
        else:
            # Show the board
            display_board(the_board)
            #Choose a position
            position = player_choice(the_board)
            #Place the marker on the position
            place_marker(the_board,player2_marker,position)
            
            
            #Check if they won
            
            if win_check(the_board,player2_marker):
                display_board(the_board)
                print('PLAYER 2 HAS WON!!')
                game_on = False
            else:
                if full_board_check(the_board):
                    display_board(the_board)
                    print("TIE GAME!!")
                    game_on = False
                else:
                    turn = 'Player 1'
            
         
    
    
       ### PLAYER TWO TURN
    
    
    
    


    if not replay():
        break
#BREAK OUT OF THE WHILE LOOP ON replay()


# In[ ]:





# In[ ]:





# In[ ]:




