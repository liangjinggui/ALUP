

PROMPT_BANKING = \
"""Your role is to identify the **user intent** represented in a given **query utterance** by comparing it with a provided **conversational utterance set**.  

- User Intent: The goal or purpose conveyed by a user in their interaction with an AI agent.  
- Predefined Intents: Intents that are already known and defined in the system.
- Novel Intents: Intents that are new and not previously defined in the system.

Your Task:
For this task, you will work with utterances in the banking domain. Given a **query utterance**, identify the utterance from the **conversational utterance set** that shares the **same intent** as the query utterance. Each utterance in the set represents a distinct user intent.

Important Rules:
1. **Do not guess.** If you are not absolutely certain that an utterance shares the same intent as the query utterance, **you must return "Cluster_id: -1."**  
2. **A match requires full alignment of intent.** Partial overlaps in wording or topic (e.g., similar keywords but different goals) do not count as a match.  
3. **Prioritize accuracy over matching.** It is better to return `Cluster_id: -1` than to risk a false positive.  

Examples:

## Example 1:
Conversational Utterance Set:

Cluster_id: 37. Utterance: how about disposable virtual cards?
Cluster_id: 39. Utterance: is there a charge for getting more cards?  
Cluster_id: 29. Utterance: What is the limit to disposable cards you can have?
Cluster_id: 60. Utterance: can you tell me the limit for top ups?
Cluster_id: 10. Utterance: can cards be delivered? where?
Cluster_id: 3. Utterance: what atms can i use my card?
Cluster_id: 14. Utterance: my card won't work right now.
Cluster_id: 36. Utterance: i need to know which flat currencies are supported by you.
Cluster_id: 43. Utterance: can i order a second card?
Cluster_id: 63. Utterance: why have i been charged more than once for the same transaction?
Cluster_id: 22. Utterance: i think someone is using my card without my permission!
Cluster_id: 4. Utterance: i need help finding the auto top up option.
Cluster_id: 23. Utterance: good day, in reviewing my credit card activity over the last several days, i see a repeated charge for a restaurant that i went to. the charge was put through twice, so i would like one of the charges corrected and the amount that was withdrawn to be reinstated
Cluster_id: 40. Utterance: what is the virtual card and how can i get one?
Cluster_id: 54. Utterance: what currencies do you accept to add money?
Cluster_id: 11. Utterance: i'm still waiting on my new card to be delivered!
Cluster_id: 73. Utterance: can i get a visa or mastercard?
Cluster_id: 15. Utterance: how come i was charged an extra fee when paying with the card?
Cluster_id: 12. Utterance: can i change my pin at any atm?
Cluster_id: 31. Utterance: what is the charge for exchanging currency?
Cluster_id: 13. Utterance: my phone got stolen, how can i use the app?
Cluster_id: 68. Utterance: why do i have a charge for an atm withdrawal? i thought these were free?
Cluster_id: 16. Utterance: i don't think i made this payment that is showing up
Cluster_id: 49. Utterance: my pin seems to be blocked, help me access it? 
Cluster_id: 0. Utterance: i need more assistance with how to activate my card.  
Cluster_id: 41. Utterance: i've lost my card. what can i do about that?
Cluster_id: 26. Utterance: is there something wrong with the atm? it would not let me pull cash out of my account.
Cluster_id: 9. Utterance: if my card expires next month, will i need to order a new one?
Cluster_id: 57. Utterance: are there fees for using an international card to top up?
Cluster_id: 45. Utterance: i made a payment but it's still pending
Cluster_id: 2. Utterance: how can i get my google pay top up to work?
Cluster_id: 1. Utterance: what is the appropriate age for my child to be able to open an account?
Cluster_id: 25. Utterance: the card payment is declined. please inform me why.
Cluster_id: 13. Utterance: i've obtained the card, how do i enable it on the app?
Cluster_id: 38. Utterance: where can i find my pin? i haven't gotten it yet.
Cluster_id: 18. Utterance: i contacted a seller about a refund last week but he still hasn't returned my money. i am frustrated and need these funds. please help me get my money back.
Cluster_id: 52. Utterance: how do i receive a refund for my item?
Cluster_id: 53. Utterance: why would my card payment be reverted?

Query Utterance:
How many transactions can I do with one disposable card?

Identified Utterance:
Cluster_id: 29. Utterance: what is limit of disposable cards i can make per day?

## Example 2:
Conversational Utterance Set:

Cluster_id: 47. Utterance: how long will it take for my top up to work? it's still pending
Cluster_id: 25. Utterance: please help me in this, as i was unable to make any payment by my new card,  it was getting declined repeatedly. i am not very happy with this incident.
Cluster_id: 61. Utterance: my top-up was reverted by the app.
Cluster_id: 57. Utterance: i need to know what charges i will incur for using a european card for top up.
Cluster_id: 18. Utterance: my card payment didn't work
Cluster_id: 65. Utterance: how can i transfer money using my credit card?
Cluster_id: 26. Utterance: this is the first time i've tried to get my money out of an atm and it didn't work. please give me my money
Cluster_id: 60. Utterance: is there a limit to how much i can top-up?
Cluster_id: 71. Utterance: why do i have to verify my top-up card?
Cluster_id: 14. Utterance: my card don't work.
Cluster_id: 15. Utterance: i noticed an extra fee when i paid with my card.
Cluster_id: 27. Utterance: my transfer was declined. why did this happen?
Cluster_id: 3. Utterance: what currencies do you accept for adding money?
Cluster_id: 35. Utterance: i tried and tried, but i could not complete my transfer. tell me what is wrong.
Cluster_id: 19. Utterance: i was told that withdrawing cash from the atm is free, but i'm charged a fee. why is this?
Cluster_id: 75. Utterance: i asked for a certain amount of money from the atm, but it gave me a different amount. less than what i asked for.
Cluster_id: 76. Utterance: i believe i got a wrong exchange rate when i got cash.
Cluster_id: 2. Utterance: how can i top up with apple pay?
Cluster_id: 21. Utterance: my card payment reverted back, i think.
Cluster_id: 22. Utterance: can you check about unauthorized use of my card, i think someone is using mine without my knowledge?
Cluster_id: 17. Utterance: i bought something while traveling, and the exchange rate was wrong.
Cluster_id: 46. Utterance: why would a cash withdrawal still be pending?
Cluster_id: 23. Utterance: yesterday, i made an incorrect payment to the wrong account for my rent payment. i need this addressed as soon as possible and the funds transferred to the correct account by no later than tomorrow.
Cluster_id: 12. Utterance: what atms accept my card?
Cluster_id: 4. Utterance: how do i access the auto top-up option?
Cluster_id: 45. Utterance: why is my card payment still pending? it should have gone through already.
Cluster_id: 67. Utterance: i need to transfer funds from china and quick expedition is crucial. approximately how long does it take for a transfer from china to go through?
Cluster_id: 58. Utterance: i had a cheque deposited, but i don't see my money yet?
Cluster_id: 36. Utterance: what currencies are supported?
Cluster_id: 56. Utterance: will i be charged for adding money by transfer?
Cluster_id: 24. Utterance: can i get a card in the eu?
Cluster_id: 41. Utterance: i've lost my card. what can i do about that?
Cluster_id: 20. Utterance: the app says i made a cash withdrawal that i didn't make
Cluster_id: 6. Utterance: i made a check deposit but the cash hasn't arrived in my account yet.
Cluster_id: 40. Utterance: i want to get my hands on one of those virtual cards!
Cluster_id: 39. Utterance: where do i order additional cards?
Cluster_id: 62. Utterance: i deposited money but my balance hasn't been updated yet.

Query Utterance:
i really need to top-up my card today urgently but my card keeps getting declined!! can you please resolve this problem or let me know if you have any alternatives

Identified Utterance:
Cluster_id: -1. (No utterance in the set matches the query intent.)

## Example 3:
Conversational Utterance Set:

Cluster_id: 3. Utterance: what currencies do you accept for adding money?
Cluster_id: 36. Utterance: what currencies are supported?
Cluster_id: 24. Utterance: can i get a card in the eu?
Cluster_id: 39. Utterance: where do i order additional cards?
Cluster_id: 12. Utterance: when will my card be delivered?
Cluster_id: 12. Utterance: what atms accept my card?
Cluster_id: 57. Utterance: i need to know what charges i will incur for using a european card for top up.
Cluster_id: 11. Utterance: my card hasn't arrived yet. what do i do?
Cluster_id: 32. Utterance: i need to know what your exchange rates are.
Cluster_id: 65. Utterance: how can i transfer money using my credit card?
Cluster_id: 13. Utterance: how do i link a new card in the app?
Cluster_id: 56. Utterance: will i be charged for adding money by transfer?
Cluster_id: 69. Utterance: what are the steps to verify identity?
Cluster_id: 27. Utterance: are there steps to see where my funds come from?
Cluster_id: 29. Utterance: how many uses are the disposable cards good for?
Cluster_id: 60. Utterance: is there a limit to how much i can top-up?
Cluster_id: 67. Utterance: i need to transfer funds from china and quick expedition is crucial. approximately how long does it take for a transfer from china to go through?
Cluster_id: 1. Utterance: how old do you need to be to open an account?
Cluster_id: 43. Utterance: i would like to get an account for my child.
Cluster_id: 0. Utterance: i need more assistance with how to activate my card.
Cluster_id: 9. Utterance: my card will expire next month, will i need to order a new one?
Cluster_id: 5. Utterance: how long should i have to wait before i see the transfer in my account?
Cluster_id: 25. Utterance: please help me in this, as i was unable to make any payment by my new card,  it was getting declined repeatedly. i am not very happy with this incident.
Cluster_id: 18. Utterance: how can my friend send money to me?
Cluster_id: 41. Utterance: i've lost my card. what can i do about that?
Cluster_id: 15. Utterance: i noticed an extra fee when i paid with my card.
Cluster_id: 75. Utterance: i asked for a certain amount of money from the atm, but it gave me a different amount. less than what i asked for.
Cluster_id: 38. Utterance: where is my pin?
Cluster_id: 52. Utterance: is it possible to get a refund for the item?
Cluster_id: 33. Utterance: how can i exchange currencies
Cluster_id: 37. Utterance: i made a purchase recently but i have decided that i'm not buying it and i need to receive my money back. can you please give me a refund asap. it's extremely urgent.
Cluster_id: 6. Utterance: i made a check deposit but the cash hasn't arrived in my account yet.
Cluster_id: 66. Utterance: why hasn't the transaction i did to my friend arrived yet?
Cluster_id: 40. Utterance: i want to get my hands on one of those virtual cards!
Cluster_id: 22. Utterance: can you check about unauthorized use of my card, i think someone is using mine without my knowledge?
Cluster_id: 27. Utterance: my transfer was declined. why did this happen?
Cluster_id: 14. Utterance: my card don't work.

Query Utterance:
what type of card will i receive?

Identified Utterance:
Cluster_id: -1. (No utterance in the set matches the query intent.)

Instructions:
1. **Compare the query utterance with each utterance in the conversational utterance set** by analyzing their semantic meaning. Focus on the **underlying intent** of each utterance.  
2. **Prioritize the order of the conversational utterance set.** Compare utterances from top to bottom, as utterances earlier in the set are more likely to be matches.  
3. Identify a matching utterance only if the intent is **exactly the same as the query utterance**. If there is any uncertainty or if the intents are not clearly aligned, do not make a match.  
4. If a match is found, write the **Cluster_id** and **Utterance** of the identified match.  

Your Turn:

Conversational Utterance Set:
{}

Query Utterance:
{}

Identified Utterance:
[Provide the final output here (**Cluster_id** and **Utterance** or Cluster_id: -1.) .]

"""

PROMPT_CLINC = \
"""Your role is to identify the **user intent** represented in a given **query utterance** by comparing it with a provided **conversational utterance set**.  

- User Intent: The goal or purpose conveyed by a user in their interaction with an AI agent.  
- Predefined Intents: Intents that are already known and defined in the system.
- Novel Intents: Intents that are new and not previously defined in the system.

Your Task:
For this task, you will work with utterances spanning multiple domains. Given a **query utterance**, identify the utterance from the **conversational utterance set** that shares the **same intent** as the query utterance. Each utterance in the set represents a distinct user intent.

Important Rules:
1. **Do not guess.** If you are not absolutely certain that an utterance shares the same intent as the query utterance, **you must return "Cluster_id: -1."**  
2. **A match requires full alignment of intent.** Partial overlaps in wording or topic (e.g., similar keywords but different goals) do not count as a match.  
3. **Prioritize accuracy over matching.** It is better to return `Cluster_id: -1` than to risk a false positive.  

Examples:

## Example 1:
Conversational Utterance Set:

Cluster_id: 16. Utterance: can you tell me how to change my oil please
Cluster_id: 79. Utterance: when should i get my oil changed
Cluster_id: 108. Utterance: i need an appointment for someone to check out my check engine light being on
Cluster_id: 95. Utterance: set a reminder for me to take my meds
Cluster_id: 14. Utterance: what was the date when i last had my car inspected
Cluster_id: 109. Utterance: would you please schedule a meeting room for 4:00 on thursday
Cluster_id: 58. Utterance: how can i change to new insurance plan
Cluster_id: 124. Utterance: when should my tires be changed
Cluster_id: 17. Utterance: i'll require a rental car from 3/6 - 3/19 in san jose
Cluster_id: 83. Utterance: i need to pay my electric bill
Cluster_id: 81. Utterance: i need to order checks, so can i order some new ones
Cluster_id: 53. Utterance: what are some ways to keep my credit score up
Cluster_id: 62. Utterance: hey do you know how to jump start a car battery
Cluster_id: 2. Utterance: can you set the alarm for noon
Cluster_id: 127. Utterance: put wash the dog on my to do list please
Cluster_id: 114. Utterance: let me know how to make a vacation request
Cluster_id: 110. Utterance: please send the police my location with gps
Cluster_id: 20. Utterance: can you change the way you talk to a male british voice
Cluster_id: 122. Utterance: let's get a timer going for 10 minutes
Cluster_id: 56. Utterance: what ingredients do you need to make lasagna
Cluster_id: 106. Utterance: my new job requires that i rollover my 401k
Cluster_id: 32. Utterance: my card is lost, i would like to report it
Cluster_id: 73. Utterance: tell me how to apply for a new credit card
Cluster_id: 94. Utterance: what did i list in my reminders
Cluster_id: 125. Utterance: how much pressure is in my tires
Cluster_id: 139. Utterance: where do i get w2 form
Cluster_id: 142. Utterance: what sorts of things can i ask you about
Cluster_id: 65. Utterance: i need to call sal
Cluster_id: 15. Utterance: i want you to quit the action
Cluster_id: 35. Utterance: what are the steps to set up direct deposit to my chase account
Cluster_id: 126. Utterance: are the items on my todo list listed alphabetically
Cluster_id: 36. Utterance: how do i get to the closest starbucks
Cluster_id: 138. Utterance: do i need to get any shots before i travel to morocco
Cluster_id: 92. Utterance: can you find me a recipe for salsa
Cluster_id: 135. Utterance: get me an uber to chilis
Cluster_id: 23. Utterance: can you talk slower please
Cluster_id: 140. Utterance: what is the weather outside
Cluster_id: 70. Utterance: do i have a meeting with paul today
Cluster_id: 130. Utterance: please transfer $250 from checking to savings
Cluster_id: 26. Utterance: confirm a reservation for nicole brown at diangelo's at 6:45 pm, please
Cluster_id: 13. Utterance: you can remove the wedding from my calendar for march 12, 2019
Cluster_id: 93. Utterance: how can i redeem rewards earned through my amex card
Cluster_id: 147. Utterance: who are they that you work for
Cluster_id: 74. Utterance: when is the next time off for a holiday here
Cluster_id: 85. Utterance: change my pin number for my checking account
Cluster_id: 148. Utterance: what is the company that made you
Cluster_id: 87. Utterance: do they use any special type of plug in spain that you need a converter for
Cluster_id: 41. Utterance: help locate my phone
Cluster_id: 134. Utterance: are there any interesting activities to do in austin
Cluster_id: 69. Utterance: how do you convert ounces to kilos
Cluster_id: 89. Utterance: can you set up a pto request for me for march 20th to april 12th
Cluster_id: 119. Utterance: send amy a text for me and tell her i need to meet
Cluster_id: 116. Utterance: please unlink my phone
Cluster_id: 66. Utterance: i'm not sure of that
Cluster_id: 31. Utterance: where is my location
Cluster_id: 55. Utterance: can i use margarine instead of butter
Cluster_id: 30. Utterance: can you tell me my credit score
Cluster_id: 99. Utterance: who is your boss
Cluster_id: 8. Utterance: when does the car payment come due
Cluster_id: 43. Utterance: please flip a coin, i choose heads!
Cluster_id: 5. Utterance: are you real or are you an ai
Cluster_id: 10. Utterance: could you reserve me a hotel room in philadelphia near chinatown from 3/19 through 3/22
Cluster_id: 25. Utterance: please turn your volume up
Cluster_id: 118. Utterance: please tell me a joke about dinosaurs
Cluster_id: 20. Utterance: what time is it in the greenwich timezone
Cluster_id: 22. Utterance: speak in the language english
Cluster_id: 120. Utterance: thanks for your cooperation
Cluster_id: 98. Utterance: i need to talk to someone about a fraudulent activity on my card
Cluster_id: 31. Utterance: look up when my payday is supposed to be
Cluster_id: 67. Utterance: i'm needing dinner suggestions for mexican food
Cluster_id: 21. Utterance: would it be okay to change your name to daphne
Cluster_id: 45. Utterance: can you block my chase account right away please
Cluster_id: 105. Utterance: find a virtual dice and roll it for me please
Cluster_id: 100. Utterance: revert to factory settings please
Cluster_id: 149. Utterance: that is affirmative

Query Utterance:
i want to get an appointment to get my oil changed

Identified Utterance:
Cluster_id: 108. Utterance: i need an appointment for someone to check out my check engine light being on

## Example 2:
Conversational Utterance Set:

Cluster_id: 92. Utterance: can you find me a recipe for salsa
Cluster_id: 56. Utterance: what ingredients do you need to make lasagna
Cluster_id: 102. Utterance: what are the reviews for mountain mikes
Cluster_id: 57. Utterance: what are my health insurance benefits
Cluster_id: 14. Utterance: how many calories are in a cheeseburger
Cluster_id: 110. Utterance: please send the police my location with gps
Cluster_id: 103. Utterance: do you know of any good mexican restaurants in seattle
Cluster_id: 46. Utterance: can you tell me a fun fact about elephants
Cluster_id: 12. Utterance: what do i have on my calendar for february 8
Cluster_id: 111. Utterance: list out what is on my shopping list
Cluster_id: 140. Utterance: what is the weather outside
Cluster_id: 67. Utterance: i'm needing dinner suggestions for mexican food
Cluster_id: 4. Utterance: could you tell me what is the apr for the credit card
Cluster_id: 3. Utterance: i need to know the routing number for my wells fargo account
Cluster_id: 30. Utterance: can you tell me my credit score
Cluster_id: 6. Utterance: what's the balance in my bank accounts
Cluster_id: 123. Utterance: what is the timezone of saigon
Cluster_id: 26. Utterance: how long is it safe to leave shrimp in the fridge
Cluster_id: 132. Utterance: is there travel alerts for greece
Cluster_id: 28. Utterance: what is my credit limit on my discover
Cluster_id: 59. Utterance: what is the interest rate on my money market account
Cluster_id: 80. Utterance: am i safe to go to africa
Cluster_id: 134. Utterance: are there any interesting activities to do in austin
Cluster_id: 55. Utterance: can i use margarine instead of butter
Cluster_id: 125. Utterance: how much pressure is in my tires
Cluster_id: 72. Utterance: how much mpg does this car get on the highway
Cluster_id: 104. Utterance: how many rewards points do i have for my visa card
Cluster_id: 94. Utterance: what did i list in my reminders
Cluster_id: 36. Utterance: how do i get to the closest starbucks
Cluster_id: 2. Utterance: can you set the alarm for noon
Cluster_id: 108. Utterance: i need an appointment for someone to check out my check engine light being on
Cluster_id: 133. Utterance: tell my bank that i'll be in canada this weekend
Cluster_id: 27. Utterance: how long do i need to cook tuna casserole for
Cluster_id: 87. Utterance: do they use any special type of plug in spain that you need a converter for
Cluster_id: 0. Utterance: water is spelled how
Cluster_id: 81. Utterance: i need to order checks, so can i order some new ones
Cluster_id: 47. Utterance: how much gas do i have in my tank
Cluster_id: 129. Utterance: whats my recent transactions on my card
Cluster_id: 71. Utterance: what is the minimum i need to pay for my cell phone bill
Cluster_id: 112. Utterance: can you put carrots on my shopping list
Cluster_id: 33. Utterance: what is the date today
Cluster_id: 10. Utterance: could you reserve me a hotel room in philadelphia near chinatown from 3/19 through 3/22
Cluster_id: 149. Utterance: that is affirmative
Cluster_id: 118. Utterance: please tell me a joke about dinosaurs
Cluster_id: 75. Utterance: go to the next song and play it
Cluster_id: 11. Utterance: is there a long wait at chili's around 5:00
Cluster_id: 26. Utterance: confirm a reservation for nicole brown at diangelo's at 6:45 pm, please
Cluster_id: 74. Utterance: when is the next time off for a holiday here
Cluster_id: 145. Utterance: where were you born
Cluster_id: 137. Utterance: what would your name be
Cluster_id: 109. Utterance: would you please schedule a meeting room for 4:00 on thursday
Cluster_id: 105. Utterance: find a virtual dice and roll it for me please
Cluster_id: 54. Utterance: what is my income this year
Cluster_id: 144. Utterance: what's the name of the current song
Cluster_id: 31. Utterance: where is my location
Cluster_id: 20. Utterance: can you change the way you talk to a male british voice
Cluster_id: 66. Utterance: i'm not sure of that
Cluster_id: 128. Utterance: how bad is the traffic on the way to downtown
Cluster_id: 70. Utterance: do i have a meeting with paul today
Cluster_id: 98. Utterance: i need to talk to someone about a fraudulent activity on my card
Cluster_id: 35. Utterance: what are the steps to set up direct deposit to my chase account
Cluster_id: 85. Utterance: change my pin number for my checking account
Cluster_id: 69. Utterance: how do you convert ounces to kilos
Cluster_id: 101. Utterance: please reserve a table for 2 at lucky's under the name melissa at 7 pm
Cluster_id: 32. Utterance: my card is lost, i would like to report it
Cluster_id: 1. Utterance: why is my bank account frozen
Cluster_id: 120. Utterance: thanks for your cooperation
Cluster_id: 68. Utterance: say what the meaning of life is
Cluster_id: 53. Utterance: what are some ways to keep my credit score up
Cluster_id: 130. Utterance: please transfer $250 from checking to savings
Cluster_id: 141. Utterance: what sort of hobbies do you enjoy
Cluster_id: 19. Utterance: what are the carry on limits for delta flights
Cluster_id: 82. Utterance: i need to track the status of my order
Cluster_id: 38. Utterance: do you have any pets and what kind

Query Utterance:
what's the nutritional info for french fries

Identified Utterance:
Cluster_id: -1. (No utterance in the set matches the query intent.)

## Example 3:
Conversational Utterance Set:

Cluster_id: 119. Utterance: send amy a text for me and tell her i need to meet
Cluster_id: 81. Utterance: i need to order checks, so can i order some new ones
Cluster_id: 97. Utterance: when will a replacement card get here
Cluster_id: 49. Utterance: a pleasure as always to speak with you, bye
Cluster_id: 65. Utterance: i need to call sal
Cluster_id: 95. Utterance: set a reminder for me to take my meds
Cluster_id: 75. Utterance: go to the next song and play it
Cluster_id: 130. Utterance: please transfer $250 from checking to savings
Cluster_id: 82. Utterance: i need to track the status of my order
Cluster_id: 133. Utterance: tell my bank that i'll be in canada this weekend
Cluster_id: 13. Utterance: you can remove the wedding from my calendar for march 12, 2019
Cluster_id: 43. Utterance: please flip a coin, i choose heads!
Cluster_id: 35. Utterance: what are the steps to set up direct deposit to my chase account
Cluster_id: 108. Utterance: i need an appointment for someone to check out my check engine light being on
Cluster_id: 110. Utterance: please send the police my location with gps
Cluster_id: 74. Utterance: when is the next time off for a holiday here
Cluster_id: 116. Utterance: please unlink my phone
Cluster_id: 15. Utterance: i want you to quit the action
Cluster_id: 135. Utterance: get me an uber to chilis
Cluster_id: 2. Utterance: can you set the alarm for noon
Cluster_id: 118. Utterance: please tell me a joke about dinosaurs
Cluster_id: 122. Utterance: let's get a timer going for 10 minutes
Cluster_id: 32. Utterance: my card is lost, i would like to report it
Cluster_id: 80. Utterance: am i safe to go to africa
Cluster_id: 136. Utterance: will you add what i'm listening to to my road trip playlist
Cluster_id: 114. Utterance: let me know how to make a vacation request
Cluster_id: 146. Utterance: turn on your whisper mode
Cluster_id: 23. Utterance: can you talk slower please
Cluster_id: 45. Utterance: can you block my chase account right away please
Cluster_id: 66. Utterance: i'm not sure of that
Cluster_id: 120. Utterance: thanks for your cooperation
Cluster_id: 22. Utterance: speak in the language english
Cluster_id: 94. Utterance: what did i list in my reminders
Cluster_id: 96. Utterance: can you please repeat that
Cluster_id: 58. Utterance: how can i change to new insurance plan
Cluster_id: 73. Utterance: tell me how to apply for a new credit card
Cluster_id: 140. Utterance: what is the weather outside
Cluster_id: 86. Utterance: can you play the rock playlist
Cluster_id: 83. Utterance: i need to pay my electric bill
Cluster_id: 139. Utterance: where do i get w2 form
Cluster_id: 36. Utterance: how do i get to the closest starbucks
Cluster_id: 109. Utterance: would you please schedule a meeting room for 4:00 on thursday
Cluster_id: 112. Utterance: can you put carrots on my shopping list
Cluster_id: 20. Utterance: can you change the way you talk to a male british voice
Cluster_id: 131. Utterance: how do you say hello in french
Cluster_id: 67. Utterance: i'm needing dinner suggestions for mexican food
Cluster_id: 92. Utterance: can you find me a recipe for salsa
Cluster_id: 76. Utterance: that’s not correct
Cluster_id: 85. Utterance: change my pin number for my checking account
Cluster_id: 105. Utterance: find a virtual dice and roll it for me please
Cluster_id: 0. Utterance: water is spelled how
Cluster_id: 111. Utterance: list out what is on my shopping list
Cluster_id: 90. Utterance: has my vacation request been approved yet
Cluster_id: 31. Utterance: look up when my payday is supposed to be
Cluster_id: 26. Utterance: confirm a reservation for nicole brown at diangelo's at 6:45 pm, please
Cluster_id: 9. Utterance: search for a flight out of la to chicago on march 3rd for under $500
Cluster_id: 149. Utterance: that is affirmative
Cluster_id: 69. Utterance: how do you convert ounces to kilos
Cluster_id: 138. Utterance: do i need to get any shots before i travel to morocco
Cluster_id: 132. Utterance: is there travel alerts for greece
Cluster_id: 127. Utterance: put wash the dog on my to do list please
Cluster_id: 50. Utterance: hello, how are you doing
Cluster_id: 17. Utterance: i'll require a rental car from 3/6 - 3/19 in san jose
Cluster_id: 53. Utterance: what are some ways to keep my credit score up
Cluster_id: 1. Utterance: why is my bank account frozen
Cluster_id: 37. Utterance: how long does it take to get to applebees in nj
Cluster_id: 12. Utterance: what do i have on my calendar for february 8
Cluster_id: 124. Utterance: when should my tires be changed
Cluster_id: 42. Utterance: what time is boarding for my flight
Cluster_id: 61. Utterance: do i need an international visa to go to tibet
Cluster_id: 144. Utterance: what's the name of the current song
Cluster_id: 68. Utterance: say what the meaning of life is
Cluster_id: 5. Utterance: are you real or are you an ai
Cluster_id: 98. Utterance: i need to talk to someone about a fraudulent activity on my card
Cluster_id: 62. Utterance: hey do you know how to jump start a car battery

Query Utterance:
send a text message for me

Identified Utterance:
Cluster_id: 119. Utterance: send amy a text for me and tell her i need to meet

Instructions:
1. **Compare the query utterance with each utterance in the conversational utterance set** by analyzing their semantic meaning. Focus on the **underlying intent** of each utterance.  
2. **Prioritize the order of the conversational utterance set.** Compare utterances from top to bottom, as utterances earlier in the set are more likely to be matches.  
3. Identify a matching utterance only if the intent is **exactly the same as the query utterance**. If there is any uncertainty or if the intents are not clearly aligned, do not make a match.  
4. If a match is found, write the **Cluster_id** and **Utterance** of the identified match.  

Your Turn:

Conversational Utterance Set:
{}

Query Utterance:
{}

Identified Utterance:
[Provide the final output here (**Cluster_id** and **Utterance** or Cluster_id: -1.) .]

"""


PROMPT_STACKOVERFLOW = \
"""Your role is to identify the **user intent** represented in a given **query utterance** by comparing it with a provided **conversational utterance set**.  

- User Intent: The goal or purpose conveyed by a user in their interaction with an AI agent.  
- Predefined Intents: Intents that are already known and defined in the system.
- Novel Intents: Intents that are new and not previously defined in the system.

Your Task:
For this task, you will work with utterances related to technical questions. Given a **query utterance**, identify the utterance from the **conversational utterance set** that shares the **same intent** as the query utterance. Each utterance in the set represents a distinct user intent.

Important Rules:
1. **Do not guess.** If you are not absolutely certain that an utterance shares the same intent as the query utterance, **you must return "Cluster_id: -1."**  
2. **A match requires full alignment of intent.** Partial overlaps in wording or topic (e.g., similar keywords but different goals) do not count as a match.  
3. **Prioritize accuracy over matching.** It is better to return `Cluster_id: -1` than to risk a false positive.  

Examples:

## Example 1:
Conversational Utterance Set:

Cluster_id: 17. Utterance: how do you get a minimal sdl program to compile and link in visual studio 2008 express?
Cluster_id: 15. Utterance: reorder/shuffle values in a row/column in excel.
Cluster_id: 7. Utterance: qt4: qabstracttablemodel drag and drop w/o mime
Cluster_id: 13. Utterance: is  svnautoversioning directive in apache subversion imporant?
Cluster_id: 19. Utterance: how to check for command key held down java/swt at startup on mac os x
Cluster_id: 14. Utterance: changing cck title for form
Cluster_id: 6. Utterance: don't mix response.write with ajax, but what about ui.htmltextwriter?
Cluster_id: 0. Utterance: wordpress - set post_date
Cluster_id: 11. Utterance: oracle 11g sql select max integer value of varchar column
Cluster_id: 18. Utterance: is it possible to use cocoa's bindings to have an editable nstableview hooked up to an nsmutablearray?
Cluster_id: 9. Utterance: how do i report progress while executing a linq expression on a large-ish data set
Cluster_id: 5. Utterance: cas policy for sharepoint application page
Cluster_id: 16. Utterance: default arguments in matlab
Cluster_id: 4. Utterance: difference between iterator and stream in scala?
Cluster_id: 8. Utterance: drupal: can i avoid forwarding after content is created / saved ?
Cluster_id: 2. Utterance: testing spring @mvc annotations
Cluster_id: 10. Utterance: when to exploit type inference in haskell?
Cluster_id: 12. Utterance: magento : manually imported product partially saved
Cluster_id: 1. Utterance: get current working directory name in bash script
Cluster_id: 3. Utterance: hibernate @onetomany - mapping to multiple join tables

Query Utterance:
how to free up a key combination in visual studio?

Identified Utterance:
Cluster_id: 17. Utterance: how do you get a minimal sdl program to compile and link in visual studio 2008 express?

## Example 2:
Conversational Utterance Set:

Cluster_id: 12. Utterance: magento : manually imported product partially saved
Cluster_id: 13. Utterance: is  svnautoversioning directive in apache subversion imporant?
Cluster_id: 7. Utterance: qt4: qabstracttablemodel drag and drop w/o mime
Cluster_id: 14. Utterance: changing cck title for form
Cluster_id: 19. Utterance: how to check for command key held down java/swt at startup on mac os x
Cluster_id: 0. Utterance: wordpress - set post_date
Cluster_id: 11. Utterance: oracle 11g sql select max integer value of varchar column
Cluster_id: 9. Utterance: how do i report progress while executing a linq expression on a large-ish data set
Cluster_id: 18. Utterance: is it possible to use cocoa's bindings to have an editable nstableview hooked up to an nsmutablearray?
Cluster_id: 17. Utterance: how do you get a minimal sdl program to compile and link in visual studio 2008 express?
Cluster_id: 6. Utterance: don't mix response.write with ajax, but what about ui.htmltextwriter?
Cluster_id: 5. Utterance: cas policy for sharepoint application page
Cluster_id: 10. Utterance: when to exploit type inference in haskell?
Cluster_id: 8. Utterance: drupal: can i avoid forwarding after content is created / saved ?
Cluster_id: 2. Utterance: testing spring @mvc annotations
Cluster_id: 4. Utterance: difference between iterator and stream in scala?
Cluster_id: 3. Utterance: hibernate @onetomany - mapping to multiple join tables
Cluster_id: 16. Utterance: default arguments in matlab
Cluster_id: 15. Utterance: reorder/shuffle values in a row/column in excel.
Cluster_id: 1. Utterance: get current working directory name in bash script

Query Utterance:
shelve in tortoisesvn?

Identified Utterance:
Cluster_id: -1. (No utterance in the set matches the query intent.)

## Example 3:
Conversational Utterance Set:

Cluster_id: 9. Utterance: how do i report progress while executing a linq expression on a large-ish data set
Cluster_id: 4. Utterance: difference between iterator and stream in scala?
Cluster_id: 14. Utterance: changing cck title for form
Cluster_id: 2. Utterance: testing spring @mvc annotations
Cluster_id: 16. Utterance: default arguments in matlab
Cluster_id: 11. Utterance: oracle 11g sql select max integer value of varchar column
Cluster_id: 3. Utterance: hibernate @onetomany - mapping to multiple join tables
Cluster_id: 7. Utterance: qt4: qabstracttablemodel drag and drop w/o mime
Cluster_id: 6. Utterance: don't mix response.write with ajax, but what about ui.htmltextwriter?
Cluster_id: 18. Utterance: is it possible to use cocoa's bindings to have an editable nstableview hooked up to an nsmutablearray?
Cluster_id: 15. Utterance: reorder/shuffle values in a row/column in excel.
Cluster_id: 13. Utterance: is  svnautoversioning directive in apache subversion imporant?
Cluster_id: 1. Utterance: get current working directory name in bash script
Cluster_id: 0. Utterance: wordpress - set post_date
Cluster_id: 12. Utterance: magento : manually imported product partially saved
Cluster_id: 17. Utterance: how do you get a minimal sdl program to compile and link in visual studio 2008 express?
Cluster_id: 8. Utterance: drupal: can i avoid forwarding after content is created / saved ?
Cluster_id: 19. Utterance: how to check for command key held down java/swt at startup on mac os x
Cluster_id: 5. Utterance: cas policy for sharepoint application page

Query Utterance:
how to match rigid types in a type class instance?

Identified Utterance:
Cluster_id: -1. (No utterance in the set matches the query intent.)

Instructions:
1. **Compare the query utterance with each utterance in the conversational utterance set** by analyzing their semantic meaning. Focus on the **underlying intent** of each utterance.  
2. **Prioritize the order of the conversational utterance set.** Compare utterances from top to bottom, as utterances earlier in the set are more likely to be matches.  
3. Identify a matching utterance only if the intent is **exactly the same as the query utterance**. If there is any uncertainty or if the intents are not clearly aligned, do not make a match.  
4. If a match is found, write the **Cluster_id** and **Utterance** of the identified match.  

Your Turn:

Conversational Utterance Set:
{}

Query Utterance:
{}

Identified Utterance:
[Provide the final output here (**Cluster_id** and **Utterance** or Cluster_id: -1.) .]

"""


