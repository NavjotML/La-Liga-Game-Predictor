import plotly.graph_objects as go
import pandas as pd

def plot_elo_evolution(df, teams):
    fig = go.Figure()
    
   
    colors = ['#A50044', '#FEBE10', '#00529F', '#6CABDD', '#EE7F00', '#004D98', '#DA291C', '#FDB913']
    
    for idx, team in enumerate(teams):
       
        home = df[df['HomeTeam'] == team][['Date', 'home_elo_after']].copy()
        home.columns = ['Date', 'elo']
        
     
        away = df[df['AwayTeam'] == team][['Date', 'away_elo_after']].copy()
        away.columns = ['Date', 'elo']
        
       
        team_data = pd.concat([home, away]).sort_values('Date').reset_index(drop=True)
        
      
        fig.add_trace(go.Scatter(
            x=list(range(1, len(team_data) + 1)),
            y=team_data['elo'].values,
            mode='lines+markers',
            name=team,
            marker=dict(size=4, color=colors[idx % len(colors)]),
            line=dict(width=2.5, color=colors[idx % len(colors)]),
            hovertemplate='<b>' + team + '</b><br>' +
                         'Match: %{x}<br>' +
                         'ELO: %{y:.0f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': 'ELO Rating Evolution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white'}
        },
        xaxis_title='Match Number',
        yaxis_title='ELO Rating',
        template='plotly_dark',
        plot_bgcolor='#0b1c2d',
        paper_bgcolor='#0b1c2d',
        hovermode='closest',
        height=550,
        font=dict(color='white', size=12),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        ),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True
        )
    )
    
    return fig