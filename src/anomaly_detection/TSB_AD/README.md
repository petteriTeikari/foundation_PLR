# TSB-AD

Due to conflict with `Moment`, easy `uv` install was not possible, so just using two models here separately from rest of the repo

```
uv add TSB-AD
  × No solution found when resolving dependencies for split (python_full_version == '3.11.*' and platform_system == 'Darwin'):
  ╰─▶ Because only momentfm<=0.1.1 is available and momentfm==0.1.1 depends on transformers==4.33.3, we can conclude that momentfm>=0.1.1 depends on
      transformers==4.33.3.
      And because all versions of tsb-ad depend on transformers>=4.38.0 and only the following versions of tsb-ad are available:
          tsb-ad==1.0
          tsb-ad==1.1
          tsb-ad==1.2
          tsb-ad==1.3
      we can conclude that momentfm>=0.1.1 and all versions of tsb-ad are incompatible.
      And because your project depends on momentfm>=0.1.1 and tsb-ad, we can conclude that your project's requirements are unsatisfiable.
  help: If this is intentional, run `uv add --frozen` to skip the lock and sync steps.

```