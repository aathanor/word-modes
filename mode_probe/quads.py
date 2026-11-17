"""Minimal quadruples for E9 (see PREREGISTRATION_v3.md). Written after the predictions were
committed and before any version-3 label existed, by the same model that wrote the code.

Each line: word | particular, one occasion | particular, in general | kind, one occasion |
kind, in general. Within a quadruple the sentences share their setting and differ in what the
noun refers to (one or more particular things, or the kind) and in whether the predicate reports
an episode or what holds in general.

python -m mode_probe.quads   -> data/control_quads.csv
"""
from __future__ import annotations

import csv

import pandas as pd

from .common import DATA, half_of

CELLS = ("P-ep", "P-gen", "K-ep", "K-gen")

QUADS = """
cat | My cat slept by the stove all through last night. | My cat sleeps by the stove every night of the winter. | Cats first reached the island on a Portuguese ship in 1650. | Cats sleep for most of the day and hunt at dusk.
cat | The cat caught a mouse in the barn this morning. | The cat on our farm catches mice in the barn. | Cats were first kept on grain ships by Phoenician traders. | Cats catch mice wherever grain is stored.
cat | Our neighbour's cat climbed the pear tree yesterday afternoon. | Our neighbour's cat climbs the pear tree whenever a dog passes. | Cats spread across Europe with Roman soldiers and traders. | Cats climb trees to escape dogs.
cat | The black cat drank the spilled milk before anyone noticed. | The black cat drinks whatever milk is left on the table. | Cats were declared sacred in Egypt during the Old Kingdom. | Cats drink little and get most of their water from food.
cat | That cat scratched my hand when I lifted it. | That cat scratches anyone who tries to lift it. | Cats were banned from the palace by the old emperor in 1640. | Cats scratch when they are held against their will.
hammer | The hammer slipped and cracked the tile on Tuesday. | This hammer slips if the handle gets wet. | Hammers with steel heads replaced bronze ones during the Iron Age. | Hammers with steel heads last for generations.
hammer | My grandfather's hammer broke when he hit the anvil. | My grandfather's hammer rings when it strikes the anvil. | Hammers were first made with claws for pulling nails in Roman times. | Hammers ring when they strike an anvil.
hammer | The hammer fell from the scaffold and landed in the mud. | The hammer hangs from a hook on the scaffold. | Hammers driven by steam appeared in the forges of the 1840s. | Hammers are kept on hooks in a tidy workshop.
hammer | The old hammer drove the last nail into the coffin. | The old hammer drives nails better than the new one. | Hammers were mass produced for the first time in Connecticut factories. | Hammers drive nails by turning a swing into a blow.
hammer | His hammer went missing from the toolbox last week. | His hammer goes missing from the toolbox every month. | Hammers became standard tools for shoemakers in the eighteenth century. | Hammers are among the oldest tools people use.
apple | The apple fell from the branch and rolled under the bench. | The apple on the top shelf stays fresh for weeks. | Apples reached North America with the first European settlers. | Apples stay fresh for months in a cool cellar.
apple | She ate the last apple in the bowl after dinner. | She eats an apple from the same bowl every evening. | Apples were introduced to England by the Romans. | Apples are eaten fresh, baked, or pressed for cider.
apple | The red apple on the windowsill was bruised by noon. | The red apple on the windowsill gets bruised by the sun. | Apples were grafted onto hardier roots for the first time by Roman growers. | Apples bruise easily when they are dropped.
apple | His apple tasted sour when he bit into it at the fair. | His apples taste sour when they are picked too early. | Apples became a major export of New Zealand in the 1890s. | Apples taste sour when they are picked too early.
apple | The apple I bought yesterday had a worm inside. | The apples from that stall always have worms inside. | Apples were blamed for a cholera outbreak in the city in 1832. | Apples sometimes have worms inside them.
river | The river flooded the lower meadow last April. | The river floods the lower meadow every April. | Rivers were first dammed for irrigation in ancient Mesopotamia. | Rivers flood their meadows when the snow melts.
river | The river froze over during the cold night. | The river freezes over in the coldest weeks of January. | Rivers were first crossed by steamboats in the 1810s. | Rivers freeze over in cold continental winters.
river | We crossed the river at the ford on our way home. | We cross the river at the ford on our way to market. | Rivers were mapped systematically for the first time in the eighteenth century. | Rivers carry silt down to the sea.
river | The river carried the broken boat past the mill that evening. | The river carries broken branches past the mill. | Rivers were first used to drive water mills in the Roman period. | Rivers carry broken branches down to the plains.
river | That river rose by two metres after the storm. | That river rises by two metres after every storm. | Rivers were first protected by pollution laws in England in 1876. | Rivers rise quickly after heavy storms.
chair | The chair creaked when he sat down at the table. | The chair creaks whenever anyone sits down at the table. | Chairs with backs became common in ordinary homes in the seventeenth century. | Chairs creak as their joints dry out.
chair | My chair tipped over when I leaned back. | My chair tips over if I lean back too far. | Chairs replaced benches in English taverns during the eighteenth century. | Chairs tip over when people lean back too far.
chair | The wooden chair stood by the window all afternoon. | The wooden chair stands by the window in every season. | Chairs made of bent wood were first produced in Vienna in the 1850s. | Chairs are placed around a table for meals.
chair | Her chair broke under the weight of the boxes. | Her chair holds the weight of the boxes she stacks on it. | Chairs with rockers were invented in North America in the eighteenth century. | Chairs hold the weight of a seated person.
chair | The chair in the hall was painted green last spring. | The chair in the hall is painted green every few years. | Chairs of moulded plastic were first sold in the 1950s. | Chairs are painted to protect the wood.
horse | The horse threw its rider at the first fence. | The horse throws its rider at the first fence. | Horses were domesticated on the steppe around 3500 BC. | Horses throw riders who pull too hard on the reins.
horse | Our horse pulled the cart up the hill this morning. | Our horse pulls the cart up the hill each morning. | Horses arrived in the Americas with Spanish ships. | Horses pull carts and ploughs on small farms.
horse | The grey horse ate the hay we left in the stall. | The grey horse eats the hay we leave in the stall. | Horses replaced oxen on most English farms during the nineteenth century. | Horses eat hay and grass and need clean water.
horse | His horse went lame on the road to the market. | His horse goes lame whenever the road is frozen. | Horses were first used to haul canal barges in the 1760s. | Horses go lame on frozen and stony ground.
horse | That horse won the race at the county fair. | That horse wins the race at the county fair every year. | Horses were first raced on a regular course in seventeenth-century England. | Horses can run at more than forty miles an hour.
bottle | The bottle rolled off the table and shattered. | This bottle rolls off the table whenever the ship lurches. | Bottles with screw caps were introduced in the 1880s. | Bottles roll easily when they are laid on their side.
bottle | The green bottle washed up on the beach this morning. | The green bottle stands on the shelf above the sink. | Bottles made of glass were first blown in Syria around the first century. | Bottles are made of glass, plastic, or metal.
bottle | His bottle of wine was opened at the wedding. | His bottle of wine is opened only at weddings. | Bottles of beer were sold in sealed form for the first time in the 1870s. | Bottles of wine are stored on their side.
bottle | The bottle I found in the cellar was still full. | The bottle in the cellar is refilled every autumn. | Bottles were first recycled under a deposit law in Oregon in 1971. | Bottles are collected for recycling in many towns.
bottle | A bottle floated past the boat as we drifted. | A bottle floats in the harbour by the steps. | Bottles were first produced by machine in Ohio in 1903. | Bottles float when they are sealed and empty.
bridge | The bridge collapsed in the gale of 1879. | The bridge sways whenever a gale blows up the estuary. | Bridges of cast iron were first built in England in the 1770s. | Bridges sway a little in strong winds.
bridge | The old bridge was closed for repairs last summer. | The old bridge is closed for repairs every summer. | Bridges were first built with stone arches by Roman engineers. | Bridges are closed for repairs from time to time.
bridge | The stone bridge carried the army across the river that day. | The stone bridge carries heavy traffic across the river. | Bridges with suspension cables spread across America after 1850. | Bridges carry roads and railways over water.
bridge | Their bridge was washed away by the flood in spring. | Their bridge is washed away by the floods every few years. | Bridges of reinforced concrete were first built in France in the 1870s. | Bridges are washed away when floods are severe.
bridge | We walked across the bridge at sunset. | We walk across the bridge at sunset most evenings. | Bridges were first lit by electric lamps in the 1880s. | Bridges are lit at night for safety.
knife | The knife slipped and cut his thumb at supper. | This knife slips and cuts his thumb if he is careless. | Knives with folding blades became popular in the seventeenth century. | Knives cut thumbs when they slip.
knife | My knife was lost in the grass during the picnic. | My knife is kept in a leather sheath on my belt. | Knives were first made of flint over two million years ago. | Knives are kept in sheaths or blocks.
knife | The kitchen knife was sharpened before the guests arrived. | The kitchen knife is sharpened before every holiday meal. | Knives of stainless steel were first made in Sheffield around 1914. | Knives need sharpening after heavy use.
knife | Her knife lay on the plate after the meal. | Her knife lies on the plate when she has finished eating. | Knives were banned from the tables of French nobles by Richelieu. | Knives are laid to the right of the plate.
knife | The knife glinted in the candlelight as he turned it. | The knife glints in the candlelight whenever it is turned. | Knives were first sold with plastic handles in the 1930s. | Knives glint when light falls on the blade.
tree | The tree fell across the road in the storm. | The tree loses its leaves early every autumn. | Trees were planted along the boulevards of Paris in the 1850s. | Trees lose their leaves in autumn.
tree | The old tree in the garden blossomed in April. | The old tree in the garden blossoms in April. | Trees from China were introduced into English gardens in the eighteenth century. | Trees blossom in spring in temperate climates.
tree | The tree by the gate was struck by lightning. | The tree by the gate is struck by lightning almost every summer. | Trees from Australia were introduced to California in the 1850s. | Trees are struck by lightning more often than open ground.
tree | My tree produced its first fruit this year. | My tree produces fruit every other year. | Trees were first planted as windbreaks on the prairies in the 1930s. | Trees produce fruit after several years of growth.
tree | The tree shaded the bench through the hot afternoon. | The tree shades the bench through every summer afternoon. | Trees were planted in city squares for shade in the nineteenth century. | Trees shade the ground beneath them.
dog | The dog barked at the postman this morning. | The dog barks at the postman every morning. | Dogs reached Australia several thousand years ago with seafaring people. | Dogs bark at strangers who come near the house.
dog | My dog found a bone in the garden yesterday. | My dog finds a bone in the garden every week. | Dogs were domesticated from wolves more than fifteen thousand years ago. | Dogs bury bones and dig them up later.
dog | Our dog slept on the porch all afternoon. | Our dog sleeps on the porch in the afternoons. | Dogs were banned from the city's parks in 1890. | Dogs sleep for about half of the day.
dog | The farmer's dog rounded up the sheep before the storm. | The farmer's dog rounds up the sheep every evening. | Dogs were first bred for herding sheep in the highlands. | Dogs can be trained to herd sheep.
dog | That dog followed me home from the station. | That dog follows me home from the station most days. | Dogs were first registered by breed in the 1870s. | Dogs follow the people they trust most closely.
fork | The fork fell off the table during dinner. | The fork sits to the left of the plate at every meal. | Forks reached English dinner tables from Italy in the seventeenth century. | Forks are laid to the left of the plate.
fork | Her silver fork was lost at the wedding. | Her silver fork is polished before every dinner party. | Forks with four tines became common in the eighteenth century. | Forks are polished to keep the silver bright.
fork | The fork bent when he stuck it into the frozen meat. | The fork bends whenever he sticks it into frozen meat. | Forks were condemned by some churchmen as a needless luxury in the 1100s. | Forks bend if they are made of soft metal.
fork | The garden fork was left out in the rain last night. | The garden fork is left out in the rain all winter. | Forks of steel replaced wooden ones on farms in the nineteenth century. | Forks are used to turn soil and lift hay.
fork | That fork scraped the plate with a loud noise. | That fork scrapes the plate with a loud noise. | Forks were introduced at the Byzantine court long before they reached the West. | Forks scrape plates if they are pressed too hard.
tomato | The tomato split open on the vine after the storm. | The tomato on the windowsill ripens slowly in the shade. | Tomatoes reached Italy from the Americas in the 1540s. | Tomatoes ripen slowly in the shade.
tomato | My tomato plant produced its first fruit last week. | My tomato plants produce fruit until the first frost. | Tomatoes were grown only as ornaments in early English gardens. | Tomatoes produce fruit until the first frost.
tomato | The tomato she sliced for the salad was perfectly ripe. | The tomatoes she grows for salads are always ripe in August. | Tomatoes were first described in a European herbal in 1544. | Tomatoes are sliced raw into salads.
tomato | The tomato he threw hit the actor on the stage. | The tomato on the counter is used for sauce every Sunday. | Tomatoes were first canned on a large scale in New Jersey in the 1840s. | Tomatoes are cooked down to make sauce.
tomato | The green tomato ripened on the shelf in a week. | The green tomatoes on the shelf ripen within a week. | Tomatoes were first sold in Paris markets around 1790. | Tomatoes ripen off the vine if kept warm.
mountain | The mountain disappeared into cloud by noon. | The mountain disappears into cloud almost every afternoon. | Mountains became a subject of landscape painting in the eighteenth century. | Mountains create their own weather on their slopes.
mountain | We climbed the mountain in a single day. | We climb the mountain every summer. | Mountains were first climbed for sport in the Alps in the 1850s. | Mountains are climbed for sport and for science.
mountain | The mountain above the village was covered in snow by October. | The mountain above the village is covered in snow by October. | Mountains were first measured by triangulation in the eighteenth century. | Mountains are covered in snow above a certain height.
mountain | That mountain shook during the earthquake. | That mountain shakes during every earthquake. | Mountains first became tourist destinations in the 1860s. | Mountains form where the plates of the crust collide.
mountain | The mountain blocked the wind from the valley that night. | The mountain blocks the wind from the valley in winter. | Mountains were protected as national parks for the first time in 1872. | Mountains block winds and cast rain shadows.
door | The door slammed shut behind her. | The door slams shut whenever the wind blows. | Doors with locks became common in English cottages in the eighteenth century. | Doors slam when the wind catches them.
door | The front door was painted blue last spring. | The front door is painted blue every few years. | Doors were fitted with letterboxes across Britain after the postal reform of 1840. | Doors are painted to protect them from the weather.
door | Our door creaked as I opened it at midnight. | Our door creaks whenever it is opened. | Doors that revolve were patented in Philadelphia in 1888. | Doors creak when their hinges are dry.
door | The kitchen door stood open all afternoon. | The kitchen door stands open in the summer. | Doors were taxed in France as a measure of wealth after 1798. | Doors are left open in hot weather.
door | His door was locked when we arrived. | His door is locked whenever he goes out. | Doors with electric locks were introduced in hotels in the 1970s. | Doors are locked to keep out thieves.
cow | The cow broke through the fence last night. | The cow breaks through the fence whenever the grass is short. | Cows were domesticated in the Near East about ten thousand years ago. | Cows break through fences when the grass is short.
cow | Our cow gave birth to a calf this morning. | Our cow gives birth to a calf each spring. | Cows reached the Americas with Spanish settlers. | Cows give birth to one calf at a time.
cow | The brown cow wandered into the village square. | The brown cow wanders into the village square most mornings. | Cows were banned from the streets of London in the 1850s. | Cows wander far if they are not fenced in.
cow | Their cow gave very little milk last winter. | Their cow gives very little milk in winter. | Cows were bred for higher milk yields during the twentieth century. | Cows give less milk in winter.
cow | That cow kicked over the bucket at milking time. | That cow kicks over the bucket at milking time. | Cows were first milked by machine in the 1890s. | Cows are milked twice a day on most dairy farms.
lamp | The lamp flickered and went out at midnight. | The lamp flickers whenever the wind gets in. | Lamps burning kerosene were introduced in the 1850s. | Lamps flicker when the wick is too short.
lamp | My lamp was knocked over by the cat. | My lamp is lit every evening at six. | Lamps lit by gas appeared in London streets in 1807. | Lamps are lit in the evening.
lamp | The oil lamp smoked badly during dinner. | The oil lamp smokes badly whenever it is turned up. | Lamps were improved by the glass chimney in the 1780s. | Lamps smoke when the wick is turned up too far.
lamp | The lamp on her desk burned all night. | The lamp on her desk burns all night. | Lamps were replaced by electric bulbs in most cities by 1920. | Lamps burn oil, gas, or electricity.
lamp | That lamp swung in the wind on the porch. | That lamp swings in the wind on the porch. | Lamps were carried by miners for the first time with a safety gauze in 1815. | Lamps swing when they hang from chains.
road | The road was blocked by snow this morning. | The road is blocked by snow every winter. | Roads were paved with tarmac for the first time in the 1900s. | Roads are blocked by snow in mountain winters.
road | The road to the village washed out in the storm. | The road to the village washes out after heavy storms. | Roads were first paved with stone slabs by the Romans. | Roads wash out where drainage is poor.
road | Our road was resurfaced last summer. | Our road is resurfaced every ten years. | Roads were first lit by gas in the 1820s. | Roads are resurfaced as they wear.
road | The coast road was closed after the accident. | The coast road is closed whenever there is fog. | Roads with toll gates spread across England in the eighteenth century. | Roads are closed when conditions are dangerous.
road | The road through the forest flooded overnight. | The road through the forest floods every spring. | Roads were improved by Macadam's method in the 1820s. | Roads flood where they cross low ground.
spoon | The spoon clattered onto the floor during supper. | The spoon rests on the saucer beside the cup. | Spoons made of silver became gifts at christenings in Tudor England. | Spoons rest on the saucer beside a cup.
spoon | Her silver spoon was stolen from the drawer. | Her silver spoon is polished every Sunday. | Spoons were first cast in pewter in the fourteenth century. | Spoons are carved from wood or made of metal.
spoon | The spoon bent in the thick porridge this morning. | The spoon bends in thick porridge. | Spoons with long handles were first made for stirring soup in the 1600s. | Spoons bend if they are made of soft metal.
spoon | His spoon slipped into the soup during the speech. | His spoon slips into the soup whenever he talks. | Spoons of stainless steel were first made in Sheffield in the 1920s. | Spoons are used for soup and dessert.
spoon | That spoon was left in the jam jar overnight. | That spoon is left in the jam jar. | Spoons of plastic were first mass produced in the 1940s. | Spoons are left in jars of honey and jam.
stone | The stone struck the window and cracked the glass. | The stone in the garden wall loosens every winter. | Stones were first cut to build walls in the Neolithic. | Stones loosen as frost works into the cracks.
stone | The stone he threw skipped five times across the pond. | The stone at the edge of the pond skips well on calm days. | Stones were first polished into tools in the Neolithic. | Stones skip well when they are flat.
stone | A heavy stone rolled down the slope last night. | A heavy stone rolls down the slope after every storm. | Stones were first quarried with iron tools in the Iron Age. | Stones roll down slopes after heavy rain.
stone | The stone by the gate was moved by the farmer. | The stone by the gate is moved every spring by the plough. | Stones were first used to mark field boundaries in Bronze Age Britain. | Stones are cleared from fields before ploughing.
stone | That stone was polished by the river over centuries. | That stone is polished by the river. | Stones were first used as standard weights in Babylon around 2000 BC. | Stones are polished smooth by running water.
"""


def quad_frame() -> pd.DataFrame:
    rows = []
    q = 0
    for line in QUADS.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        w, sents = parts[0], parts[1:]
        assert len(sents) == 4, line
        for cell, s in zip(CELLS, sents):
            rows.append({"qid": q, "word": w, "cell": cell, "sentence": s})
        q += 1
    df = pd.DataFrame(rows)
    df["half"] = df["word"].map(half_of)
    df.insert(0, "sid", [f"q{i:04d}" for i in range(len(df))])
    return df


def write_quads():
    df = quad_frame()
    df.to_csv(DATA / "control_quads.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    return df


if __name__ == "__main__":
    print(write_quads().groupby("word").size().to_string())
