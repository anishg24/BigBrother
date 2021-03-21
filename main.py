from keras.models import load_model
import numpy as np
import pickle
import discord
from json import load, loads
import DiscordUtils
from keras_preprocessing.sequence import pad_sequences
from datetime import datetime
import requests
from discord.ext import commands
import zulu

bot = commands.Bot(command_prefix="!")

rating_index = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
max_input_length = 1403
model = load_model("model/model.h5")

with open("model/tokenizer.pkl", "rb") as t:
    tokenizer = pickle.load(t)

with open("config.json", "r") as c:
    config = load(c)


def process_message_for_data(sentence):
    tokens = []
    for word in sentence.lower().split():
        try:
            tokens.append(tokenizer.word_index[word])
        except KeyError:
            tokens.append(0)
    transformed_sentence = pad_sequences([tokens], maxlen=max_input_length, padding="post")
    predictions = model.predict(transformed_sentence)
    prediction = predictions[0]
    return np.argmax(prediction), np.amax(prediction), predictions


async def handle_toxicity(message):
    index, val, preds = process_message_for_data(message.content)
    if val > config["THRESHOLD"]:
        try:
            await message.delete()
            await message.channel.send(
                f"Hey! Big Brother here! {message.author.display_name} remember to be respectful to others!")
            log_embed = discord.embeds.Embed(
                title="Log Report",
                timestamp=datetime.now(),
                color=discord.Color.red()
            )
            log_embed.add_field(
                name="User",
                value=f"Name: {message.author}\nID: {message.author.id}", inline=False
            )
            log_embed.add_field(
                name="Message Sent",
                value=f"||{message.content}||", inline=False
            )
            log_embed.add_field(
                name="Model Info",
                value=f"`toxic`: {str(val * 100)[:4]}%\n"
                      f"Threshold: {config['THRESHOLD'] * 100}%", inline=False
            )
            await message.guild.get_channel(config["LOG_CHANNEL"]).send(embed=log_embed)
            return
        except Exception as e:
            print("ERROR DELETING!")
            print(e)


@bot.command(name="factcheck", aliases=["fc", "check"])
async def fact_check(ctx, *args):
    API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    query = ' '.join(args).lower()
    params = {"key": config["GOOGLE_API"], "query": query, "languageCode": "en-US"}
    response = loads(requests.get(API, params=params).text)
    if not response:
        await ctx.send(f"I am unable to check that fact!")
    embeds = []
    for i, claim in enumerate(response['claims']):
        review = claim["claimReview"][0]
        publisher = review["publisher"]
        url = review["url"]
        date = zulu.parse(review["reviewDate"]).datetime
        rating = review["textualRating"]

        embed = discord.Embed(title=f"Source #{i + 1}", color=discord.Color.dark_gold(), timestamp=date)
        embed.add_field(name="Fact Checker Info: ", value=f"Name: {publisher['name']}\nWebsite: {publisher['site']}", inline=False)
        embed.add_field(name="Rating: ", value=rating, inline=False)
        embed.add_field(name="More Info: ", value=url, inline=False)

        embeds.append(embed)
    await ctx.send(f"**{len(embeds)}** sources found!")
    paginator = DiscordUtils.Pagination.AutoEmbedPaginator(ctx, timeout=30, remove_reactions=True)
    await paginator.run(embeds)


@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    await handle_toxicity(message)
    await bot.process_commands(message)


bot.run(config["TOKEN"])
