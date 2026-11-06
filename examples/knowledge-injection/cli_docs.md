```bash
trainsay 1.0.0 — haul your text across the terminal inside an ASCII freight car

USAGE:
    trainsay [OPTIONS] [MESSAGE]...

DESCRIPTION:
    trainsay prints an ASCII steam locomotive pulling a freight car across
    your terminal window, with your message stenciled directly onto the side
    of the cargo container. If no [MESSAGE] is given, it reads from stdin.

OPTIONS:
    -c, --car <TYPE>
        Choose the type of railcar carrying your message:
          boxcar   Enclosed metal container with sliding doors
          tanker   Cylindrical liquid carrier with warning placards
          flatbed  Open timber flatbed exposing the raw text payload
        [default: boxcar]

    -s, --speed <LEVEL>
        Set the locomotive throttle and smoke stack intensity:
          1   Idle; light steam puffs (o o o)
          2   Cruising; thick rolling smoke (@ @ @)
          3   Full throttle; heavy billowing smoke and sparks (* * *)
        [default: 2]

    -d, --derail
        Cause the locomotive to hit a broken rail, spilling cargo blocks
        across the screen and exiting with status code 1.

EXAMPLES:
    trainsay --car boxcar --speed 1 "Deploying to production"
        OUTPUT:
              o       ______
              \_    //======\\
             _(_)_  || CU   ||
           O|o o o|_||______||__________________________
          / and ___ |                                   |
         /_/---/^\\-|   DEPLOYING TO PRODUCTION         |
        (""")  ("" ")`-----------------------------------'
         --O----O--   o  o  o  o  o  o  o  o  o  o  o  o

    uptime | trainsay --car tanker --speed 3
        OUTPUT:
              o       ______
              \_    //======\\
             _(_)_  || HAZ  ||
           O|* * *|_||______||__________________________
          / and ___ |                                   |
         /_/---/^\\-|   up 3 days, load: 0.85         |
        (""")  ("" ")`-----------------------------------'
         --O----O--   *  *  *  *  *  *  *  *  *  *  *  *

    trainsay --car flatbed --speed 2 --derail "Merge conflict in main.rs"
        OUTPUT:
               \ \ \
                \ \ \
                 _..._
                /     \      ___________________________
               | (@) (@)|   /                           \
               |   __   |--<  MERGE CONFLICT IN MAIN.RS  |
               \  \__/  /   \___________________________/
              /`--------`\
             / / / / / / /\\  <--- [BOILER EXPLOSION]
            -----------------~~~~~~~~~~~~~

EXIT STATUS:
    0   Train arrived safely at the destination terminal.
    1   Train derailed (--derail was passed or track failure occurred).
```