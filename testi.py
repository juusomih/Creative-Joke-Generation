from espeakng import ESpeakNG

esng = ESpeakNG(voice="en-us")
ipa = esng.g2p ('Hello World!')
print(ipa)

