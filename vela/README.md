To apply the helm chart on the settings file, please follow the steps included [here](https://github.com/project-codeflare/mlbatch/tree/main/tools/pytorchjob-generator).

Note: some parameters cannot be applied from the helm chart, such as the `GIT_PAT`, so you may have to use the file `n1_g8_geo-vlm_phi3.yaml` as a reference and manually add items from there.


Create a manifest file using a helm chart:
```
helm template -f settings_single_node.yaml mlbatch/pytorchjob-generator > n1_g8_geo-vlm_phi3.yaml
```

To launch the AppWrapper:
```
oc apply -f n1_g8_geo-vlm_phi3.yaml         
```