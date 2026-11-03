# Pi Weight-Space Programming (π · WSP) Terminal Demo
# Experiment 1: "Which game was this neural network trained on?"
# Recorded with Charm VHS (https://github.com/charmbracelet/vhs)

Output "docs/video/demo.mp4"
Output "docs/video/demo.gif"

Set Shell "zsh"
Set FontSize 14
Set Width 1320
Set Height 1040
Set WindowBar "Colorful"
Set BorderRadius 10
Set Theme "Catppuccin Mocha"
Set Padding 18
Set TypingSpeed 55ms

# 1. Start Pi in hidden mode so recording begins with Pi open & banner active
Hide
Type "export PATH=/Users/fcpenha/.local/bin:/opt/homebrew/bin:$PATH"
Enter
Type "cd /Users/fcpenha/git/pi-weight-space-programming"
Enter
Type "clear"
Enter
Type "pi -e extensions/index.ts --wsp-exp 1 --thinking low"
Enter
Sleep 3500ms
Show

# 2. Pause on the banner so the viewer can take in the interface
Sleep 2500ms

# 3. Natural Language Prompt from Paper Experiment 1
Type "Which game was this neural network trained on? Here are the observations from the weights: Channel 0 has large weights strictly at row 8; Channels 1 and 4 have velocity pairing across rows 1-8; Channel 2 has reward weights at rows 1-3; and only left and right actions are active."
Sleep 1500ms
Enter

# 4. Real live run: Pi processes the prompt and streams the mechanistic deduction
Sleep 30s
