To run                                                    
                                                                                                                                                                         
  # 1. Install dependencies
  pip install -r requirements.txt                                                                                                                                        
                                                            
  # 2. Build the graph (first time only, ~2 min)                                                                                                                         
  python3 graph/map_loader.py
  python3 graph/enrichment.py                                                                                                                                             
                                                            
  # 3. Test full CLI pipeline                                                                                                                                            
  python3 main.py --src Shahbagh --dst "Gulshan 1" --preset balanced
                                                                                                                                                                         
  # 4. Launch the UI                                        
  streamlit run app/streamlit_app.py
