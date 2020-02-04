import torch

infersent = torch.load('encoder/infersent.allnli.pickle', map_location=lambda storage, loc: storage)
infersent.use_cuda = False

infersent.set_glove_path("../../tools/glove.840B/glove.840B.300d.txt")
infersent.build_vocab_k_words(K=100000)

#sentence = 'A man plays an instrument.'
#sentence = "The Rachel Corrie , named after an American activist killed in 2003 as she tried to prevent an Israeli bulldozer from razing a Palestinian home , had been due to join the other boats in the flotilla last week but was delayed by technical problems ."
#sentence = "Hamas , which rejects Israel 's existence , won Palestinian parliamentary elections in early 2006 ."
#sentence = "When Hamas militants seized an Israeli soldier , Gilad Shalit , in a raid that June , Israel further reduced what was permitted in and out of the coastal territory ."
#sentence = "A year later , after Hamas fighters drove the more moderate Fatah movement from Gaza , Israel imposed a full closure on Gaza , permitting in only basic humanitarian goods ."
sentence = "They made 66 percent of their shots ."


sentence = "I'm gonna put the microphone down while I help them get into the boat."
sentence = "Mom needs dialysis. Clinic flooded. No cable internet or phone. Cell service spotty.  Who do I call?"
sentence = "Smh i would choose to switch internet companies at the right time. No wifi is killin me."
sentence = "Day 2- tweeting on a computer. No Phone."
sentence = "Harvey Day 4: the rain continues. We still have power, no internet. Outside, few stores are open. Some people are openly looting food."
sentence = "If your home has been damaged dm me my dad and I remodel houses we'll give FREE estimates."
sentence = "Everything is ruined. Please pray the rain stops!"
sentence = "Holy crap! Beltway 8 has collapsed next to my house!"
sentence = "Houston Texas is flooding Crazy! This Weather is destroying everything, We hope to find peace!"
sentence = "Even if you are far from the devastation of Hurricane Harvey, there are ways to contribute."

infersent.visualize(sentence, tokenize=True)
