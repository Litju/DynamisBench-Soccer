# M4 grader qualification protocol

The qualification gate requires independently authored analytical or synthetic controls
held outside the checkout until after this protocol is committed. The control author
chooses expected decisions independently of the grader. Keep reference values and
submissions out of the report and out of the candidate workspace.

| Control class | Minimum controls | Maximum failures |
| --- | ---: | ---: |
| Correct scientific answers | 2 | 0 false rejections |
| Plausible scientific errors | 2 | 0 false acceptances |
| Adversarial cases | 2 | 0 false acceptances |
| Refusal and abstention | 2 | 0 incorrect decisions |
| Real positive task-specific admissions | 1 | 0 false rejections |
| Negative task-specific admissions | 1 | 0 false acceptances |
| Repeated control evaluation | All controls | 0 differences |
| Candidate isolation and checkpoint release | Full offline suite | 0 failures |

Report the numerator and denominator for each class, observed grader finding categories,
and the independently verified task-admission state. A missing class blocks the gate.
Known-answer checks of the comparison function establish only diagnostic evidence.
Qualification also requires a genuinely independently reviewed engineering-admitted
reference to pass through the formal offline runner, with a privileged oracle and
weaker stub baselines. A blank review catalog establishes zero positive admissions.
Engineering acceptance requires final-head review and exact-head Linux CI.
